"""Forecasters compared in the revision: SeismoGPT and the baselines.

Every forecaster answers the same question in the same units:

    given the context tokens   ctx  [T_ctx, C, K]   (the first T_ctx tokens of the
                                                     window that starts at P,
                                                     anchor-normalised, x 10)
    return the next n tokens   pred [n, C, K]

`baselines/evaluate.py` passes ONLY the context to `forecast`, so no forecaster can
see the future it is asked to predict. Everything else (items, context length,
metrics, CSV) is shared, so the numbers of different forecasters are directly
comparable.

    name         trained on       idea
    -----------  ---------------  -----------------------------------------------
    zero         -                the ground returns to rest (reference: skill 0)
    persistence  -                repeat the last context token
    ar_burg      the context      linear prediction fitted to this trace (Burg)
    linear       training set     one linear map: last P tokens -> next token
    gru          training set     recurrent network, same tokens as SeismoGPT
    knn          training set     copy the continuation of the most similar
                                  training context (retrieval / look-up)
    knn_params   training set     the same, most similar in source/receiver parameters
                                  (an oracle: uses the true parameters)
    seismogpt    training set     the transformer of the paper
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import lfilter, lfiltic

ROOT = Path(__file__).resolve().parent.parent          # repository root
for p in (ROOT, ROOT / "quality_assurance"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ======================================================================================
# Common interface
# ======================================================================================
class Forecaster:
    """Base class. Subclasses implement `forecast`; `forecast_many` may be overridden
    when a forecaster is faster on many items at once (kNN)."""

    name = "base"

    def forecast(self, ctx: torch.Tensor, n: int) -> torch.Tensor:
        """ctx [T_ctx, C, K] -> prediction [n, C, K] (float32, CPU)."""
        raise NotImplementedError

    def forecast_many(self, contexts: list[torch.Tensor], ns: list[int],
                      metas: list[dict] | None = None) -> list[torch.Tensor]:
        """metas: the metadata row of each item. Only the parameter-space look-up
        (knn_params) reads it; every other model ignores it."""
        return [self.forecast(c, n) for c, n in zip(contexts, ns)]


# ======================================================================================
# 1. Zero: the trivial reference. NCC = 0, skill = 0, SNR = 0 dB by definition.
# ======================================================================================
class Zero(Forecaster):
    name = "zero"

    def forecast(self, ctx, n):
        return torch.zeros((n,) + tuple(ctx.shape[1:]), dtype=torch.float32)


# ======================================================================================
# 2. Persistence: "tomorrow looks like today". Repeat the last context token.
# ======================================================================================
class Persistence(Forecaster):
    name = "persistence"

    def forecast(self, ctx, n):
        return ctx[-1:].float().repeat(n, 1, 1)


# ======================================================================================
# 3. Per-trace linear prediction (Burg). No training data: an autoregressive model
#    x[t] = -sum_k a[k] x[t-k] is fitted to the context of each channel and run forward.
#    Burg's method (Burg 1967, a geophysics classic) always returns a stable filter,
#    so the free-running forecast cannot blow up.
# ======================================================================================
def burg(x: np.ndarray, order: int) -> np.ndarray:
    """AR coefficients a = [1, a1, ..., ap] of x by Burg's method.

    Forward and backward prediction errors ef, eb start as the signal itself. At each
    order the reflection coefficient k minimises their summed power; |k| < 1, which
    keeps the filter stable. The coefficients follow from the Levinson recursion.
    """
    ef = x.astype(np.float64).copy()
    eb = ef.copy()
    a = np.ones(1)
    for _ in range(order):
        f, b = ef[1:], eb[:-1]                       # align e_f(t) with e_b(t-1)
        den = f @ f + b @ b
        if den <= 0.0:                               # silent context: nothing to fit
            break
        k = -2.0 * (f @ b) / den
        ef, eb = f + k * b, b + k * f
        a = np.append(a, 0.0)
        a = a + k * a[::-1]                          # Levinson update
    return a


class ARBurg(Forecaster):
    name = "ar_burg"

    def __init__(self, order: int = 32):
        self.order = order

    def forecast(self, ctx, n):
        T, C, K = ctx.shape
        x = ctx.permute(1, 0, 2).reshape(C, T * K).double().numpy()     # [C, samples]
        p = max(1, min(self.order, x.shape[1] // 3))                    # short contexts: lower order
        out = np.zeros((C, n * K))
        for c in range(C):
            a = burg(x[c], p)
            past = x[c, ::-1][: len(a) - 1]                             # most recent sample first
            zi = lfiltic([1.0], a, past)                                # filter state = the past
            out[c] = lfilter([1.0], a, np.zeros(n * K), zi=zi)[0]       # run forward with no input
        return torch.from_numpy(out).float().reshape(C, n, K).permute(1, 0, 2).contiguous()


# ======================================================================================
# 4. Learned linear autoregression. One linear map, fitted to the training set by
#    least squares (baselines/train_linear.py): the last P tokens of all channels
#    (zero-padded before P: the ground is at rest) -> the next token. Rolled out
#    token by token, exactly like SeismoGPT. "Is a non-linear model needed at all?"
# ======================================================================================
def linear_features(tokens: torch.Tensor, P: int) -> torch.Tensor:
    """tokens [B, T, C, K] -> features [B, T, P*C*K + 1] for predicting token t+1 from
    tokens t-P+1..t (left zero padding, plus a constant 1 for the bias)."""
    B, T, C, K = tokens.shape
    flat = tokens.reshape(B, T, C * K)
    padded = torch.cat([flat.new_zeros(B, P - 1, C * K), flat], dim=1)  # [B, T+P-1, CK]
    win = padded.unfold(1, P, 1)                                         # [B, T, CK, P]
    win = win.permute(0, 1, 3, 2).reshape(B, T, P * C * K)               # oldest token first
    return torch.cat([win, flat.new_ones(B, T, 1)], dim=-1)


class LinearAR(Forecaster):
    name = "linear"

    def __init__(self, weights: str):
        w = np.load(weights)
        self.W = torch.from_numpy(w["W"]).float()        # [P*C*K + 1, C*K]
        self.P = int(w["P"])

    def forecast(self, ctx, n):
        T, C, K = ctx.shape
        hist = torch.cat([ctx.new_zeros(self.P, C, K), ctx.float()])      # zero history before P
        preds = []
        for _ in range(n):
            f = torch.cat([hist[-self.P:].reshape(-1), torch.ones(1)])   # same layout as training
            nxt = (f @ self.W).reshape(1, C, K)
            preds.append(nxt)
            hist = torch.cat([hist, nxt])
        return torch.cat(preds)


# ======================================================================================
# 5. GRU: recurrent network on the same tokens, trained with the same data module
#    (baselines/gru.py, configs/baseline_gru.yaml). The hidden state carries the past,
#    so each rollout step costs one GRU step.
# ======================================================================================
class GRUForecaster(Forecaster):
    name = "gru"

    def __init__(self, checkpoint: str, device: str = "cpu"):
        from baselines.gru import GRULightning
        lm = GRULightning.load_from_checkpoint(checkpoint, map_location=device)
        self.net = lm.net.eval()
        self.amp_compress = bool(lm.hparams.get("amp_compress", False))  # same option as SeismoGPT
        self.device = device

    @torch.no_grad()
    def forecast(self, ctx, n):
        from models.lightning_module import compress_amplitude, expand_amplitude
        x = ctx.float().unsqueeze(0).to(self.device)        # [1, T, C, K]
        if self.amp_compress:
            x = compress_amplitude(x)
        y, h = self.net(x)                                   # read the whole context once
        nxt = y[:, -1:]                                      # first future token
        preds = [nxt]
        for _ in range(n - 1):
            nxt, h = self.net(nxt, h)                        # one step, state carried on
            preds.append(nxt)
        out = torch.cat(preds, dim=1)[0].cpu()
        return expand_amplitude(out) if self.amp_compress else out


# ======================================================================================
# 6. Nearest-neighbour retrieval. The library (baselines/build_knn_library.py) holds
#    training windows, normalised like the test windows. For a context of T_ctx tokens,
#    find the training window whose first T_ctx tokens are closest (Euclidean), and
#    return its next n tokens. With flip=True the library also counts with reversed
#    polarity (SeismoGPT is trained with polarity flips too). k > 1 averages the k best.
#    This answers: "does the model do more than look up a similar training trace?"
# ======================================================================================
class KNN(Forecaster):
    name = "knn"

    def __init__(self, library: str, k: int = 1, flip: bool = True,
                 device: str = "cpu", chunk: int = 50_000):
        lib = Path(library)
        self.tokens = np.load(lib / "tokens.npy", mmap_mode="r")      # [N, T_lib, C, K]
        # cumulative energy per token: ||first t tokens||^2 = cum[:, t-1], for any t at once
        self.cum = np.load(lib / "cum_energy.npy", mmap_mode="r")     # [N, T_lib]
        self.k, self.flip, self.device, self.chunk = k, flip, device, chunk
        self.last_neighbours = None                                     # (index, sign) per item

    def forecast(self, ctx, n):
        return self.forecast_many([ctx], [n])[0]

    @torch.no_grad()
    def forecast_many(self, contexts, ns, metas=None):
        N, T_lib, C, K = self.tokens.shape
        T_q = torch.tensor([c.shape[0] for c in contexts])
        need = int((T_q + torch.tensor(ns)).max())
        if need > T_lib:
            raise ValueError(f"library windows have {T_lib} tokens, but a context + forecast "
                             f"needs {need}; rebuild with a larger --max_tokens")
        best_i = self._search(contexts, T_q, metas)        # [B, k] signed: -(i+1) = flipped
        self.last_neighbours = best_i
        preds = []
        for b in range(len(contexts)):
            t0, n = int(T_q[b]), ns[b]
            acc = np.zeros((n, C, K))
            for j in best_i[b]:
                sign = -1.0 if j < 0 else 1.0
                idx = -j - 1 if j < 0 else j
                acc += sign * self.tokens[idx, t0:t0 + n]                # the neighbour's continuation
            preds.append(torch.from_numpy(acc / self.k).float())
        return preds

    @torch.no_grad()
    def _search(self, contexts, T_q, metas):
        """The k nearest library windows of every context (waveform distance)."""
        N, T_lib, C, K = self.tokens.shape
        B = len(contexts)
        T_max, D = int(T_q.max()), C * K
        dev = self.device

        # queries zero-padded to T_max tokens: the padding adds nothing to q . l
        Q = torch.zeros(B, T_max * D)
        for b, c in enumerate(contexts):
            Q[b, : c.numel()] = c.float().reshape(-1)
        Q = Q.to(dev)
        q2 = (Q * Q).sum(1, keepdim=True)                                 # [B, 1]
        t_idx = (T_q - 1).to(dev)

        best_d = torch.full((B, self.k), float("inf"), device=dev)
        best_i = torch.zeros((B, self.k), dtype=torch.long, device=dev)  # signed: -(i+1) = flipped
        for s in range(0, N, self.chunk):
            L = torch.from_numpy(np.ascontiguousarray(self.tokens[s:s + self.chunk, :T_max]))
            L = L.reshape(L.shape[0], -1).float().to(dev)                 # [n_c, T_max*D]
            l2 = torch.from_numpy(np.ascontiguousarray(self.cum[s:s + self.chunk])).float().to(dev)
            l2 = l2[:, t_idx].T                                           # [B, n_c]: energy of first T_q tokens
            dot = Q @ L.T                                                 # [B, n_c]  (the only big operation)
            cand_d, cand_i = [q2 + l2 - 2 * dot], [torch.arange(s, s + L.shape[0], device=dev).expand(B, -1)]
            if self.flip:                                                 # ||q + l||^2: library with sign reversed
                cand_d.append(q2 + l2 + 2 * dot)
                cand_i.append(-(cand_i[0] + 1))
            d = torch.cat([best_d] + cand_d, 1)
            i = torch.cat([best_i] + cand_i, 1)
            best_d, pos = d.topk(self.k, dim=1, largest=False)
            best_i = i.gather(1, pos)
        return best_i.cpu().numpy()


# ======================================================================================
# 6b. Nearest neighbour in PARAMETER space, as Reviewer 1 phrased it: the training trace
#     whose source and receiver are most similar (distance, depth, azimuth, focal
#     mechanism). This is an oracle: it uses the true parameters of the test event,
#     which SeismoGPT never sees. Its continuation is the "Green's-function replay" of
#     the closest training example. Magnitude is left out: after normalisation it only
#     sets the amplitude. Each group (distance, depth, azimuth, mechanism) gets equal
#     weight after standardisation. Reversing the mechanism (M -> -M) reverses the
#     waveform exactly, so flip=True also searches -M with the sign of the copy reversed.
# ======================================================================================
MT_COLS = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]


def source_features(rows) -> np.ndarray:
    """[N, 10] = distance, depth, cos az, sin az, unit moment tensor (6).
    rows: a DataFrame or a list of dicts (metadata rows)."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    m = df[MT_COLS].to_numpy(np.float64)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)           # shape of the mechanism only
    az = np.deg2rad(df["azimuth_deg"].to_numpy(np.float64))
    return np.column_stack([df["distance_deg"], df["src_depth_km"], np.cos(az), np.sin(az), m])


class KNNParams(KNN):
    name = "knn_params"
    GROUPS = [[0], [1], [2, 3], [4, 5, 6, 7, 8, 9]]            # distance | depth | azimuth | mechanism

    def __init__(self, library, **kw):
        super().__init__(library, **kw)
        f = source_features(pd.read_csv(Path(library) / "index.csv"))
        self.mu, self.sd = f.mean(0), f.std(0) + 1e-12
        self.w = np.ones(f.shape[1])
        for g in self.GROUPS:                                  # each group counts once in total
            self.w[g] = 1.0 / np.sqrt(len(g))
        self.lib_f = torch.from_numpy(self._scale(f)).float().to(self.device)

    def _scale(self, f):
        return (f - self.mu) / self.sd * self.w

    @torch.no_grad()
    def _search(self, contexts, T_q, metas):
        if metas is None:
            raise ValueError("knn_params needs the metadata of the test items")
        q = source_features(metas)
        queries = [(q, 1)]
        if self.flip:                                          # mechanism reversed: -M, copy * -1
            qf = q.copy()
            qf[:, 4:] *= -1.0
            queries.append((qf, -1))
        best_d = best_i = None
        for qq, sign in queries:
            d = torch.cdist(torch.from_numpy(self._scale(qq)).float().to(self.device), self.lib_f)
            i = torch.arange(d.shape[1], device=d.device).expand_as(d)
            i = i if sign > 0 else -(i + 1)
            if best_d is not None:
                d, i = torch.cat([best_d, d], 1), torch.cat([best_i, i], 1)
            best_d, pos = d.topk(self.k, dim=1, largest=False)
            best_i = i.gather(1, pos)
        return best_i.cpu().numpy()


# ======================================================================================
# 7. SeismoGPT, through the KV-cache rollout of quality_assurance/efficient_rollout.py,
#    so its numbers are identical to those of that script.
# ======================================================================================
class SeismoGPT(Forecaster):
    name = "seismogpt"

    def __init__(self, checkpoint: str, device: str = "cpu"):
        from efficient_rollout import CachedGPT
        from models.lightning_module import GPTLightning
        lm = GPTLightning.load_from_checkpoint(checkpoint, map_location=device)
        self.amp_compress = bool(getattr(lm, "amp_compress", False))   # v2 option
        self.cached = CachedGPT(lm.gpt.to(device).eval())
        self.device = device

    def forecast(self, ctx, n):
        from efficient_rollout import rollout_kvcache
        T, C, K = ctx.shape
        # rollout_kvcache reads the future only in teacher mode; in free mode the
        # appended zeros only set the length
        x = torch.cat([ctx.float(), ctx.new_zeros(n, C, K).float()]).unsqueeze(0).to(self.device)
        _, pred, _, _ = rollout_kvcache(self.cached, x, T, n, K, mode="free",
                                        amp_compress=self.amp_compress)        # [n*K, C]
        return torch.from_numpy(pred).reshape(n, K, C).permute(0, 2, 1).contiguous()


# ======================================================================================
def build(name: str, args) -> Forecaster:
    """Create a forecaster from the command-line arguments of evaluate.py."""
    if name == "zero":
        return Zero()
    if name == "persistence":
        return Persistence()
    if name == "ar_burg":
        return ARBurg(order=args.ar_order)
    if name == "linear":
        return LinearAR(args.weights)
    if name == "gru":
        return GRUForecaster(args.checkpoint, args.device)
    if name == "knn":
        return KNN(args.library, k=args.knn_k, flip=not args.knn_no_flip, device=args.device)
    if name == "knn_params":
        return KNNParams(args.library, k=args.knn_k, flip=not args.knn_no_flip, device=args.device)
    if name == "seismogpt":
        return SeismoGPT(args.checkpoint, args.device)
    raise ValueError(f"unknown model {name}")


MODELS = ["zero", "persistence", "ar_burg", "linear", "gru", "knn", "knn_params", "seismogpt"]
