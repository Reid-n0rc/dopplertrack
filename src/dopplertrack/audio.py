"""Audio decoding and whitened low-band spectrograms."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.signal as ss


def load_audio(path: str | Path, sr: int = 4000) -> tuple[np.ndarray, int]:
    """Decode any ffmpeg-readable file to mono float32 at `sr` Hz."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr),
           "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy(), sr


@dataclass
class Spectrogram:
    t: np.ndarray      # (T,) centre time of each frame, s
    f: np.ndarray      # (F,) bin frequencies, Hz
    S: np.ndarray      # (F, T) whitened power, dB above per-bin median

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    @property
    def df(self) -> float:
        return float(self.f[1] - self.f[0])

    def sample(self, t, f, fill=0.0):
        """Bilinear sample at (t, f) pairs (arrays); outside the grid -> fill."""
        t = np.asarray(t, float); f = np.asarray(f, float)
        ti = (t - self.t[0]) / self.dt
        fi = (f - self.f[0]) / self.df
        ok = (ti >= 0) & (ti <= len(self.t) - 1) & (fi >= 0) & (fi <= len(self.f) - 1) & np.isfinite(ti) & np.isfinite(fi)
        out = np.full(t.shape, fill, float)
        ti = ti[ok]; fi = fi[ok]
        i0 = np.floor(ti).astype(int); j0 = np.floor(fi).astype(int)
        i1 = np.minimum(i0 + 1, len(self.t) - 1); j1 = np.minimum(j0 + 1, len(self.f) - 1)
        wt = ti - i0; wf = fi - j0
        v = (self.S[j0, i0] * (1 - wt) * (1 - wf) + self.S[j0, i1] * wt * (1 - wf)
             + self.S[j1, i0] * (1 - wt) * wf + self.S[j1, i1] * wt * wf)
        out[ok] = v
        return out

    def crop(self, t0, t1, f0=None, f1=None) -> "Spectrogram":
        m = (self.t >= t0) & (self.t <= t1)
        fm = np.ones_like(self.f, bool)
        if f0 is not None: fm &= self.f >= f0
        if f1 is not None: fm &= self.f <= f1
        return Spectrogram(self.t[m], self.f[fm], self.S[np.ix_(fm, m)])

    def save(self, path):
        np.savez_compressed(path, t=self.t, f=self.f, S=self.S.astype(np.float32))

    @classmethod
    def load(cls, path) -> "Spectrogram":
        z = np.load(path)
        return cls(z["t"], z["f"], z["S"])


def compute_spectrogram(x: np.ndarray, sr: int, nperseg: int = 8192, hop_s: float = 0.25,
                        fmin: float = 40.0, fmax: float = 450.0,
                        whiten_window_s: float | None = 600.0, chunk_frames: int = 16384) -> Spectrogram:
    """High-resolution STFT in the low band, converted to dB and whitened.

    Whitening subtracts, per frequency bin, the running median over
    `whiten_window_s` (or the global median if None). This flattens mains
    hum and stationary machinery so only *changing* tones stand out.
    """
    hop = max(1, round(hop_s * sr))
    # frames k = 0 .. K-1 cover samples [k*hop, k*hop + nperseg); computed in chunks so a
    # many-hour recording never holds its full complex STFT in memory
    K = (len(x) - nperseg) // hop + 1
    f = np.fft.rfftfreq(nperseg, 1 / sr)
    band = (f >= fmin) & (f <= fmax)
    f = f[band]
    parts = []
    for k0 in range(0, K, chunk_frames):
        k1 = min(k0 + chunk_frames, K)
        seg = x[k0 * hop:(k1 - 1) * hop + nperseg]
        _, _, Z = ss.stft(seg, sr, nperseg=nperseg, noverlap=nperseg - hop, window="hann",
                          boundary=None, padded=False)
        parts.append((10 * np.log10(np.abs(Z[band]) ** 2 + 1e-20)).astype(np.float32))
    L = np.concatenate(parts, axis=1)
    t = (nperseg / 2 + np.arange(K) * hop) / sr
    if whiten_window_s is None:
        L = L - np.median(L, axis=1, keepdims=True)
    else:
        w = int(whiten_window_s / (hop / sr))
        if w >= L.shape[1]:
            L = L - np.median(L, axis=1, keepdims=True)
        else:
            # block-wise running median (cheap, adequate for slow drifts)
            nb = int(np.ceil(L.shape[1] / w))
            med = np.zeros_like(L)
            centers = []
            for b in range(nb):
                sl = slice(b * w, min((b + 1) * w, L.shape[1]))
                med[:, sl] = np.median(L[:, sl], axis=1, keepdims=True)
                centers.append((sl.start + sl.stop) / 2)
            # linearly blend block medians to avoid seams
            cm = np.array([np.median(L[:, b * w:min((b + 1) * w, L.shape[1])], axis=1) for b in range(nb)])  # (nb, F)
            idx = np.arange(L.shape[1])
            med = np.stack([np.interp(idx, centers, cm[:, k]) for k in range(L.shape[0])])
            L = L - med
    return Spectrogram(t, f, L.astype(np.float32))
