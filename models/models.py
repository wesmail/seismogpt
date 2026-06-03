# Generic imports
import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Flash Attention Diagnostics
# =============================================================================

def check_flash_attention_available() -> Dict[str, bool]:
    """
    Check which SDPA backends are available.
    Call this at startup to verify Flash Attention is enabled.
    """
    info = {
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        # Flash Attention requires compute capability >= 8.0 (Ampere+)
        major, minor = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{major}.{minor}"
        info["supports_flash_attention"] = major >= 8
    return info


def print_flash_attention_status():
    """Pretty print Flash Attention availability."""
    info = check_flash_attention_available()
    print("=" * 60)
    print("SDPA / Flash Attention Status")
    print("=" * 60)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    if info.get("supports_flash_attention") and info.get("flash_sdp_enabled"):
        print("✅ Flash Attention SHOULD be active for bf16/fp16 without explicit mask")
    else:
        print("⚠️  Flash Attention may NOT be available - check GPU & PyTorch version")
    print()

# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from the RoFormer paper.
    Applies rotation to query and key vectors based on their position,
    enabling relative position awareness without explicit position embeddings.

    Cache is fully pre-allocated at __init__ time up to max_len as registered
    buffers (fp32). This avoids any runtime reallocation, Python-level non-buffer
    tensors, and dtype-conversion surprises during mixed-precision training.
    The forward pass is a simple slice + dtype cast — no conditional logic.
    """
    def __init__(self, dim: int, max_len: int = 10000, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        # inv_freq: [dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin tables up to max_len in fp32.
        # Shape: [1, 1, max_len, dim]  (ready to broadcast over B and num_heads)
        # Memory: 2 × max_len × dim × 4 bytes
        #   e.g. max_len=5000, dim=32  →  2 × 5000 × 32 × 4 = 1.28 MB — negligible.
        t = torch.arange(max_len).float()                     # [max_len]
        freqs = torch.outer(t, inv_freq)                      # [max_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)               # [max_len, dim]
        self.register_buffer("cos_cache", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)  # [1,1,max_len,dim]
        self.register_buffer("sin_cache", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)  # [1,1,max_len,dim]

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k: (B, num_heads, seq_len, head_dim)
        Returns:
            Rotated (q, k) with same shape.
        """
        seq_len = q.size(2)
        # Slice to actual seq_len and cast to match q/k dtype (e.g. bf16).
        # No reallocation — just a view + dtype cast (cheap).
        cos = self.cos_cache[:, :, :seq_len, :].to(q.dtype)  # [1, 1, seq_len, dim]
        sin = self.sin_cache[:, :, :seq_len, :].to(q.dtype)  # [1, 1, seq_len, dim]
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        cos = cos[..., : x1.size(-1)]
        sin = sin[..., : x1.size(-1)]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# =============================================================================
# Transformer Encoder Layer with RoPE (Flash Attention optimized)
# =============================================================================

class RoPETransformerEncoderLayer(nn.Module):
    """
    Pre-norm transformer encoder layer with RoPE applied to Q, K in attention.
    
    OPTIMIZATION: Uses F.scaled_dot_product_attention with is_causal=True
    and NO explicit mask, allowing Flash Attention to activate.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model and self.head_dim % 2 == 0, \
            "d_model must be divisible by nhead; head_dim must be even for RoPE"

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.attn_dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            src: Input tensor of shape [B, T, d_model]
            is_causal: If True, applies causal masking via SDPA's is_causal flag
                       (NO explicit mask - this enables Flash Attention)
        """
        # Pre-norm self-attention
        x = self.self_attn_norm(src)
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = self.w_k(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        # CRITICAL: Do NOT pass attn_mask when is_causal=True
        # This allows Flash Attention to be used
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # ← No explicit mask!
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal,  # ← SDPA handles causal masking internally
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        src = src + self.dropout(self.w_o(attn_out))
        
        # Pre-norm FFN
        src = src + self.ffn(self.ffn_norm(src))
        return src


# =============================================================================
# Multi-Resolution STFT Loss
# =============================================================================

class MultiResSTFTLoss(nn.Module):
    """
    Multi-resolution STFT magnitude L1 loss.
    """
    def __init__(self, n_ffts: Tuple[int, ...] = (256, 1024, 4096), eps: float = 1e-8):
        super().__init__()
        self.n_ffts = n_ffts
        self.eps = eps
        # Pre-register hann windows as buffers to avoid recreation each forward
        for n_fft in n_ffts:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft), persistent=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: [B, C, T]
        """
        B, C, T = x.shape
        loss = x.new_zeros(())
        
        for n_fft in self.n_ffts:
            hop = n_fft // 4
            win = getattr(self, f"window_{n_fft}").to(dtype=torch.float32)

            x_ = x.reshape(B * C, T)
            y_ = y.reshape(B * C, T)

            X = torch.stft(x_, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                           window=win, center=True, return_complex=True)
            Y = torch.stft(y_, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                           window=win, center=True, return_complex=True)

            magX = (X.abs() + self.eps)
            magY = (Y.abs() + self.eps)

            loss = loss + (magX - magY).abs().mean()
        
        return (loss / float(len(self.n_ffts))).to(x.dtype)


# =============================================================================
# Per-token causal CNN embedding (within-token axis K)
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    """
    Within-token CNN + attention pooling over K.

    Input:  x [B, T, C, K]
    Output: e [B, T, d_model]

    - No causality over K (full token is observed context).
    - Dilated convs with symmetric padding keep length K unchanged.
    - Attention pooling learns which within-token positions matter.
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kernel_size: int = 7,
        num_layers: int = 4,
        dilation_growth: int = 2,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        attn_dropout: float = 0.0,
        use_last: bool = True,
    ):
        super().__init__()
        assert kernel_size >= 2
        assert num_layers >= 1

        self.d_model = d_model
        self.use_last = use_last
        self.kernel_size = int(kernel_size)
        self.dilation_growth = int(dilation_growth)

        self.in_proj = nn.Conv1d(in_channels, d_model, kernel_size=1, bias=False)

        blocks = []
        for i in range(num_layers):
            d = self.dilation_growth ** i  # 1,2,4,8,...
            pad = (d * (self.kernel_size - 1)) // 2  # symmetric "same" padding for odd kernels

            blocks.append(
                nn.ModuleDict(dict(
                    conv=nn.Conv1d(
                        d_model, d_model,
                        kernel_size=self.kernel_size,
                        dilation=d,
                        padding=pad,
                        bias=False,
                    ),
                    norm=nn.GroupNorm(1, d_model),
                    act=act if isinstance(act, nn.Module) else act(),
                    drop=nn.Dropout(dropout),
                    pw=nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
                ))
            )
        self.blocks = nn.ModuleList(blocks)

        # Attention pooling over K
        self.pool_scorer = nn.Linear(d_model, 1, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)

        # Projection after pooling (optionally concat with last)
        in_dim = (2 * d_model) if use_last else d_model
        self.summary_proj = nn.Linear(in_dim, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x as [B, T, C, K], got {x.shape}")

        B, T, C, K = x.shape

        # [B*T, C, K]
        h = x.reshape(B * T, C, K)
        h = self.in_proj(h)  # [B*T, d_model, K]

        # Non-causal residual blocks over within-token time K
        for blk in self.blocks:
            y = blk["conv"](h)   # symmetric padding keeps length
            y = blk["norm"](y)
            y = blk["act"](y)
            y = blk["drop"](y)
            y = blk["pw"](y)
            h = h + y

        # [B*T, d_model, K] -> [B*T, K, d_model]
        h_k = h.transpose(1, 2).contiguous()

        # Attention weights over K
        logits = self.pool_scorer(h_k).squeeze(-1)  # [B*T, K]
        w = torch.softmax(logits, dim=-1)           # [B*T, K]
        w = self.attn_drop(w)

        # Weighted sum: [B*T, d_model]
        tok_att = (h_k * w.unsqueeze(-1)).sum(dim=1)

        if self.use_last:
            tok_last = h[..., -1]  # [B*T, d_model]
            tok_cat = torch.cat([tok_att, tok_last], dim=-1)
        else:
            tok_cat = tok_att

        tok = self.summary_proj(tok_cat)
        tok = self.out_norm(tok)

        return tok.view(B, T, self.d_model)

class TokenEmbeddingFast(nn.Module):
    """
    Fast token embedding:
      x: [B, T, C, K] -> e: [B, T, d_model]

    Steps:
      1) 1x1 conv mixes channels at each k (no dilations, no within-token temporal conv)
      2) pool over K: mean, and optionally last sample
      3) linear projection + LayerNorm

    Constructor matches :class:`TokenEmbedding` so callers (CLI/config, GPT) can swap
    implementations without changing kwargs. ``kernel_size``, ``num_layers``,
    ``dilation_growth``, and ``attn_dropout`` are ignored by this path.
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kernel_size: int = 7,
        num_layers: int = 4,
        dilation_growth: int = 2,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        attn_dropout: float = 0.0,
        use_last: bool = True,
    ):
        super().__init__()
        self.use_last = use_last

        # channel mix per sample
        self.in_proj = nn.Conv1d(in_channels, d_model, kernel_size=1, bias=False)
        self.act = act if isinstance(act, nn.Module) else act()
        self.drop = nn.Dropout(dropout)

        in_dim = (2 * d_model) if use_last else d_model
        self.out_proj = nn.Linear(in_dim, d_model, bias=True)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [B,T,C,K], got {x.shape}")

        B, T, C, K = x.shape
        h = x.reshape(B * T, C, K)          # [B*T, C, K]
        h = self.in_proj(h)                # [B*T, d_model, K]
        h = self.act(h)
        h = self.drop(h)

        # pool over within-token axis K
        tok_mean = h.mean(dim=-1)          # [B*T, d_model]

        if self.use_last:
            tok_last = h[..., -1]          # [B*T, d_model]
            tok = torch.cat([tok_mean, tok_last], dim=-1)  # [B*T, 2*d_model]
        else:
            tok = tok_mean                 # [B*T, d_model]

        tok = self.out_proj(tok)           # [B*T, d_model]
        tok = self.out_norm(tok)
        return tok.view(B, T, -1)      


# =============================================================================
# GPT Model (Flash Attention Optimized)
# =============================================================================

class GPT(nn.Module):
    """
    GPT model for seismic tokens.
    Uses TokenCausalEmbedding over K (within-token time) instead of 1x1 conv "Embedding".

    Probabilistic checkpoints use ``shared_mu_head`` / ``shared_sigma_head`` with a shared
    ``horizon_embed``. The second half of the 2×C output is a nonnegative scale read out by
    ``GPTLightning``: Gaussian σ (``time_loss=="nll"``), Laplace scale b (``nll_laplace``),
    or Student-t scale s (``nll_studentt``)—same head, same clamp semantics in Lightning.
    When fine-tuning into a probabilistic mode, load with ``strict=False`` if new sigma-head
    weights should stay randomly initialized.
    
    OPTIMIZATIONS:
    - Flash Attention enabled via is_causal flag (no explicit mask)
    - Optional torch.compile() support
    - Pre-registered buffers for STFT windows
    """

    def __init__(
        self,
        in_channels: int = 3,
        kernel_size: int = 16,
        num_tokens: int = 256,
        d_model: int = 128,
        num_heads: int = 2,
        num_enc_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 5000,
        dim_feedforward_multiplier: int = 4,
        # Token CNN embed config
        token_cnn_kernel: int = 7,
        token_cnn_layers: int = 4,
        token_cnn_dilation_growth: int = 2,
        token_cnn_dropout: float = 0.0,

        # ── Multi-step prediction horizons ──────────────────────────────────
        # Horizon-conditioned shared MLP heads (+Embedding); idx 0 = inference horizon.
        num_pred_horizons: int = 1,
        # If True, shared sigma MLP and concat [mu, scale] per sample (2*C channels).
        # Lightning interprets scale as σ / b / s depending on ``time_loss``.
        probabilistic_output: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_tokens = num_tokens

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_enc_layers = num_enc_layers
        self.dropout = dropout
        self.max_len = max_len
        self.dim_feedforward = self.d_model * dim_feedforward_multiplier

        self.num_pred_horizons = max(1, int(num_pred_horizons))
        self.probabilistic_output = bool(probabilistic_output)
        self.out_channels = self.in_channels * (2 if self.probabilistic_output else 1)

        ck = self.in_channels * self.kernel_size
        self.horizon_embed = nn.Embedding(self.num_pred_horizons, self.d_model)
        nn.init.normal_(self.horizon_embed.weight, std=0.02)
        self.shared_mu_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, ck),
        )
        self.shared_sigma_head: Optional[nn.Sequential]
        if self.probabilistic_output:
            self.shared_sigma_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, ck),
            )
            # Conservative final-layer init: small initial scales for Gaussian σ, Laplace b,
            # and Student-t s (same ``shared_sigma_head`` for all probabilistic losses).
            nn.init.normal_(self.shared_sigma_head[-1].weight, std=0.01)
            nn.init.constant_(self.shared_sigma_head[-1].bias, -2.0)
        else:
            self.shared_sigma_head = None

        # Causal CNN over K inside each token (time-only; no frequency branch)
        self.time_token_embed = TokenEmbeddingFast(
            in_channels=in_channels,
            d_model=self.d_model,
            kernel_size=token_cnn_kernel,
            num_layers=token_cnn_layers,
            dilation_growth=token_cnn_dilation_growth,
            dropout=token_cnn_dropout,
            act=nn.GELU(),
        )

        # Stack of RoPE encoder layers (Flash Attention optimized)
        self.encoder_layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=self.d_model,
                nhead=num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_enc_layers)
        ])

        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x_time: torch.Tensor, is_causal: bool = False):
        """
        Args:
            x_time:    [B, T, C, K]
            is_causal: enable causal masking in SDPA (Flash Attention compatible)

        Returns:
            predictions : List[Tensor[B, T*K, C_out]], length = num_pred_horizons
                C_out = C (deterministic) or 2*C (concat [mu, scale] per channel; scale is σ, b,
                or s per ``GPTLightning.time_loss``).
                predictions[0]  horizon-1 (next token), used at inference
                predictions[h]  horizon h+1 (training-only auxiliary heads)

        Note: when num_pred_horizons == 1 the list has exactly one element.
              Always use predictions[0] at inference.
        """
        if x_time.dim() != 4:
            raise ValueError(f"Expected 4D input [B, T, C, K], got {x_time.shape}")

        B, T, C, K = x_time.shape
        if C != self.in_channels or K != self.kernel_size:
            raise ValueError(
                f"Input has C={C},K={K} but model expects C={self.in_channels},K={self.kernel_size}"
            )

        # ── Encoder (time-only) ─────────────────────────────────────────────
        h = self.time_token_embed(x_time)       # [B, T, d_model]
        h = self.dropout_layer(h)

        for layer in self.encoder_layers:
            h = layer(h, is_causal=is_causal)        # [B, T, d_model]

        # ── Multi-step prediction heads (horizon-conditioned shared MLP) ─────
        predictions: List[torch.Tensor] = []
        for idx in range(self.num_pred_horizons):
            hz = self.horizon_embed.weight[idx].view(1, 1, -1)  # [1,1,d_model]
            h_cond = h + hz
            mu_h = self.shared_mu_head(h_cond)                       # [B, T, C*K]
            mu_h = mu_h.view(B, T, K, self.in_channels).contiguous()

            if self.probabilistic_output and self.shared_sigma_head is not None:
                raw_sig = self.shared_sigma_head(h_cond)
                raw_sig = raw_sig.view(B, T, K, self.in_channels)
                sigma_h = F.softplus(raw_sig) + 1e-4
                mu_flat = mu_h.reshape(B, T * K, self.in_channels)
                sig_flat = sigma_h.reshape(B, T * K, self.in_channels)
                out_h = torch.cat([mu_flat, sig_flat], dim=-1)             # [B, T*K, 2*C]
            else:
                out_h = mu_h.reshape(B, T * K, self.in_channels)           # [B, T*K, C]

            predictions.append(out_h)

        return predictions


# =============================================================================
# Utility: Compile model with torch.compile() for extra speedup
# =============================================================================

def compile_model(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
    """
    Wrap model with torch.compile() for additional speedup.
    
    Args:
        model: The model to compile
        mode: Compilation mode. Options:
            - "default": Good balance of compile time and speedup
            - "reduce-overhead": Best for small batches / inference
            - "max-autotune": Slower compile, potentially faster runtime
    
    Returns:
        Compiled model (or original if torch.compile unavailable)
    """
    if hasattr(torch, "compile"):
        print(f"Compiling model with mode='{mode}'...")
        return torch.compile(model, mode=mode)
    else:
        print("torch.compile() not available (requires PyTorch 2.0+)")
        return model