import numpy as np
from scipy.signal import welch, correlate, hilbert

def compute_snr_db(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
    """Signal-to-noise ratio in dB. Inputs are 1-D or 2-D (flattened internally)."""
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    signal_power = (y_true ** 2).mean().item()
    noise_power = ((y_pred - y_true) ** 2).mean().item()
    return 10.0 * np.log10(signal_power / (noise_power + eps))


def normalized_xcorr(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Normalized cross-correlation (scalar cosine similarity). Inputs flattened."""
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    num = np.sum(y_true * y_pred)
    den = np.sqrt(np.sum(y_true ** 2) * np.sum(y_pred ** 2)) + eps
    return (num / den).item()


def compute_psd_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fs: float = 1.0,
    eps: float = 1e-12,
) -> dict:
    """
    Welch PSD comparison for a single 1-D signal pair.
    Returns dict: psd_logl2, psd_rel.
    """
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    L = len(y_true)
    nperseg = min(1024, L)
    if nperseg < 8:
        nperseg = L
    noverlap = nperseg // 2
    f, Pt = welch(y_true, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="density")
    _, Pp = welch(y_pred, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="density")
    log_Pt = np.log10(Pt + eps)
    log_Pp = np.log10(Pp + eps)
    psd_logl2 = np.mean((log_Pt - log_Pp) ** 2).item()
    psd_rel = np.mean(np.abs(Pt - Pp) / (Pt + eps)).item()
    return {"psd_logl2": psd_logl2, "psd_rel": psd_rel}

def best_shift_by_xcorr(y_true, y_pred, fs, max_shift_s=10):
    """
    Find integer sample lag (pred shifted) that maximizes correlation.
    Returns (best_lag_samples, y_pred_shifted).
    Positive lag means shift y_pred forward (delay), i.e. y_pred_shifted[t] = y_pred[t - lag]
    """
    # flatten per channel or use 1D flattened vector
    y_t = y_true.ravel()
    y_p = y_pred.ravel()
    maxlag = int(max_shift_s * fs)
    corr = correlate(y_t, y_p, mode="full")
    lags = np.arange(-len(y_p) + 1, len(y_t))
    # restrict lags:
    center = np.where((lags >= -maxlag) & (lags <= maxlag))[0]
    if len(center) == 0:
        best_idx = np.argmax(corr)
    else:
        best_idx = center[np.argmax(corr[center])]
    best_lag = lags[best_idx]
    # shift by best_lag (positive lag => shift pred right)
    if best_lag > 0:
        y_p_shifted = np.pad(y_pred, ((best_lag, 0), (0, 0)), mode="constant")[: len(y_pred)]
    elif best_lag < 0:
        y_p_shifted = np.pad(y_pred, ((0, -best_lag), (0, 0)), mode="constant")[-best_lag: len(y_pred) - best_lag]
    else:
        y_p_shifted = y_pred.copy()
    return int(best_lag), y_p_shifted

def amplitude_scale_ls(y_true, y_pred, eps=1e-12):
    """
    Fit scalar a that minimizes ||y_true - a*y_pred||^2 (closed form).
    Works on flattened arrays or per-channel separately if needed.
    Returns a (scalar) and scaled_pred = a*y_pred.
    """
    num = np.sum(y_true.ravel() * y_pred.ravel())
    den = np.sum(y_pred.ravel() ** 2) + eps
    a = num / den
    return a, (a * y_pred)

def envelope_signal(x):
    """Return analytic envelope via Hilbert transform. x: 1D or 2D (T, C)."""
    if x.ndim == 1:
        return np.abs(hilbert(x))
    else:
        return np.abs(hilbert(x, axis=0))

# Example combined robust SNR computation
def compute_robust_snr(y_true, y_pred, fs, max_shift_s=5, low_energy_thresh=1e-6):
    """
    Returns dict with:
      - raw_snr_db
      - shifted_snr_db (after best-lag align)
      - scaled_snr_db (after amplitude scaling)
      - scaled_shifted_snr_db (scale + align)
      - envelope_snr_db (on envelopes)
      - note: may return None for some if signal energy too small
    """
    out = {}
    raw = compute_snr_db(y_true, y_pred)
    out["snr_raw_db"] = raw

    sig_power = (y_true.ravel() ** 2).mean()
    if sig_power < low_energy_thresh:
        out["note"] = "low_energy"
        return out

    # best shift
    lag, ypred_shift = best_shift_by_xcorr(y_true, y_pred, fs, max_shift_s=max_shift_s)
    out["best_lag_samples"] = lag
    out["snr_shifted_db"] = compute_snr_db(y_true, ypred_shift)

    # amplitude scale
    a, ypred_scaled = amplitude_scale_ls(y_true, y_pred)
    out["amp_scale_a"] = float(a)
    out["snr_scaled_db"] = compute_snr_db(y_true, ypred_scaled)

    # scaled + shifted
    _, ypred_scaled_shift = best_shift_by_xcorr(y_true, ypred_scaled, fs, max_shift_s=max_shift_s)
    out["snr_scaled_shifted_db"] = compute_snr_db(y_true, ypred_scaled_shift)

    # envelope SNR
    env_t = envelope_signal(y_true)
    env_p = envelope_signal(y_pred)
    out["snr_envelope_db"] = compute_snr_db(env_t, env_p)
    return out
