"""
Helpers for analyzing HARP XWAV humpback whale recordings with STUMPY.

* ``read_xwav`` memory-maps the samples of an XWAV (RIFF/WAVE + ``harp`` chunk) file
* ``matlab_mfcc`` is a numpy port of MATLAB's ``mfcc(audioIn, fs)`` with default settings
* ``compute_pan_mp`` is a resumable, cached wrapper around ``stumpy.stimp``
* ``motif_strength``, ``top_pan_motifs`` and ``refine_motifs`` rank motifs across lengths

Run as a script to compute the MFCC and the pan matrix profile in a terminal::

    python whale_mfcc.py --excl-denom 1 --step 50
"""

import argparse
import os
import struct
import sys
import time

import numpy as np
from scipy.fft import dct, rfft
from scipy.signal import get_window

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import stumpy  # noqa: E402
from stumpy import config  # noqa: E402

WAV_PATH = os.path.join(HERE, "SOCAL_H_76_230427_014845.x.wav")
MFCC_CACHE = os.path.join(HERE, "whale_mfcc_matlab.npz")
REALMIN = np.finfo(np.float64).tiny


def read_xwav(path):
    """
    Return ``(samples, fs)`` where ``samples`` is a read-only int16 memmap of the
    ``data`` chunk. Non-audio chunks such as ``harp`` are skipped.
    """
    chunks = {}
    with open(path, "rb") as f:
        riff, _, wave = struct.unpack("<4sI4s", f.read(12))
        if riff != b"RIFF" or wave != b"WAVE":
            raise ValueError(f"{path} is not a RIFF/WAVE file")
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            chunk_id, size = struct.unpack("<4sI", header)
            chunks[chunk_id] = (f.tell(), size)
            if chunk_id == b"data":
                break
            f.seek(size + (size & 1), 1)
        f.seek(chunks[b"fmt "][0])
        audio_format, n_channels, fs, _, _, bits = struct.unpack("<HHIIHH", f.read(16))

    if audio_format != 1 or n_channels != 1 or bits != 16:
        raise ValueError(
            f"Expected mono 16-bit PCM, got format={audio_format}, "
            f"channels={n_channels}, bits={bits}"
        )
    offset, size = chunks[b"data"]
    samples = np.memmap(path, dtype="<i2", mode="r", offset=offset, shape=(size // 2,))
    return samples, fs


def matlab_band_edges(fs=None):
    """
    The 42 default ``BandEdges`` of MATLAB's ``mfcc`` (Slaney's design): 13 linearly
    spaced edges starting at 133.33 Hz, followed by 29 log-spaced edges up to ~6.86 kHz.
    Edges above ``fs / 2`` are dropped, e.g. at 2 kHz only 14 edges (12 bands) remain.
    """
    linear = 133.3333333333 + 66.6666666667 * np.arange(13)
    log = linear[-1] * 1.0711703 ** np.arange(1, 30)
    edges = np.concatenate([linear, log])
    if fs is not None:
        edges = edges[edges <= fs / 2]
    return edges


def mel_filter_bank(band_edges, fs, n_fft):
    """
    Half-overlapped triangular filters; every edge except the first and last is also
    the center frequency of a neighboring filter. Returns ``(n_bands, n_bins)``.
    """
    n_bins = int(np.ceil(band_edges[-1] * n_fft / fs)) + 1
    freqs = np.arange(n_bins) * fs / n_fft
    n_bands = len(band_edges) - 2
    fb = np.zeros((n_bands, n_bins))
    for b in range(n_bands):
        lo, center, hi = band_edges[b : b + 3]
        rising = (freqs - lo) / (center - lo)
        falling = (hi - freqs) / (hi - center)
        fb[b] = np.clip(np.minimum(rising, falling), 0.0, None)
        fb[b] /= (hi - lo) / 2  # bandwidth normalization
    return fb


def matlab_mfcc(
    x,
    fs,
    window_sec=0.03,
    overlap_sec=0.02,
    num_coeffs=13,
    band_edges=None,
    chunk_frames=5000,
    hop_sec=None,
    dtype=np.float64,
    verbose=True,
):
    """
    Port of MATLAB's ``coeffs = mfcc(audioIn, fs)`` with its default parameters.

    Returns an ``(n_frames, 1 + num_coeffs)`` array in MATLAB's column order: column 0
    is the log energy (MATLAB column 1, ``LogEnergy="append"``) and columns
    ``1 ... num_coeffs`` are the cepstral coefficients c0, c1, ... (MATLAB columns 2+).
    Frames are not centered: frame ``k`` covers samples ``[k * hop, k * hop + win)``.
    ``hop_sec`` (if given) overrides ``overlap_sec``. Like MATLAB, ``num_coeffs`` is
    capped at the number of mel bands.
    """
    win_len = int(round(window_sec * fs))
    if hop_sec is None:
        hop = win_len - int(round(overlap_sec * fs))
    else:
        hop = int(round(hop_sec * fs))
    n_frames = (len(x) - win_len) // hop + 1
    window = get_window("hamming", win_len, fftbins=True).astype(np.float32)
    if band_edges is None:
        band_edges = matlab_band_edges(fs)
    fb = mel_filter_bank(band_edges, fs, win_len).astype(np.float32)
    n_bins = fb.shape[1]
    if n_bins > win_len // 2 + 1:
        raise ValueError("The filter bank extends beyond the Nyquist frequency")
    num_coeffs = min(num_coeffs, fb.shape[0])

    out = np.empty((n_frames, 1 + num_coeffs), dtype=dtype)
    t0 = time.time()
    for f0 in range(0, n_frames, chunk_frames):
        f1 = min(n_frames, f0 + chunk_frames)
        seg = np.asarray(x[f0 * hop : (f1 - 1) * hop + win_len], dtype=np.float32)
        seg /= 32768.0  # audioread scaling of int16 to [-1, 1)
        frames = np.lib.stride_tricks.sliding_window_view(seg, win_len)[::hop]
        energy = np.einsum("ij,ij->i", frames, frames, dtype=np.float64)
        spec = np.abs(rfft(frames * window, n=win_len, axis=1, workers=-1)[:, :n_bins])
        mel = spec @ fb.T
        log_mel = np.log(np.maximum(mel.astype(np.float64), REALMIN))
        out[f0:f1, 0] = np.log(np.maximum(energy, REALMIN))
        out[f0:f1, 1:] = dct(log_mel, type=2, norm="ortho", axis=1)[:, :num_coeffs]
        if verbose:
            print(f"\rMFCC {f1:,}/{n_frames:,} frames ({time.time() - t0:.0f} s)", end="")
    if verbose:
        print()
    return out


def load_or_compute_mfcc(wav_path=WAV_PATH, cache_path=MFCC_CACHE, verbose=True):
    """Returns ``(coeffs, frame_rate)``, cached to ``cache_path``"""
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["coeffs"], float(cached["frame_rate"])
    x, fs = read_xwav(wav_path)
    coeffs = matlab_mfcc(x, fs, verbose=verbose)
    hop = int(round(0.03 * fs)) - int(round(0.02 * fs))
    frame_rate = fs / hop
    np.savez(cache_path, coeffs=coeffs, frame_rate=frame_rate)
    return coeffs, frame_rate


WAV_10H_PATH = os.path.join(HERE, "SOCAL_H_76_230426_193500_df100.x.x.wav")
FEATURE_FS = 2000  # the 10 h file is the 200 kHz recording decimated x100
FEATURE_WINDOW_SEC = 0.1
FEATURE_HOP_SEC = 0.01


def decimate(x, factor, chunk=12_000_000, pad=20_000):
    """
    Anti-aliased decimation of a (memory-mapped) int16 signal by an integer
    ``factor`` with ``resample_poly``, processed in padded chunks. Keeps the int16
    amplitude scale (as float32) so ``matlab_mfcc`` can be applied unchanged.
    """
    from scipy.signal import resample_poly

    chunk -= chunk % factor
    out = []
    for start in range(0, len(x), chunk):
        stop = min(len(x), start + chunk)
        a, b = max(0, start - pad), min(len(x), stop + pad)
        seg = resample_poly(np.asarray(x[a:b], dtype=np.float32), 1, factor)
        lo = (start - a) // factor
        out.append(seg[lo : lo + (stop - start) // factor].astype(np.float32))
    return np.concatenate(out)


def load_or_compute_features(wav_path, cache_path, verbose=True):
    """
    MATLAB-style MFCC at ``FEATURE_FS`` (2 kHz, 12 mel bands from 133 Hz to 1 kHz)
    with a 0.1 s window and a 10 ms hop, i.e. 100 frames/s. Recordings at a higher
    sample rate are decimated first. Returns ``(coeffs, frame_rate)`` where the columns
    of the float32 ``coeffs`` are log energy, c0, c1, ..., c11 (MATLAB columns 1, 2, ...).
    """
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["coeffs"], float(cached["frame_rate"])
    x, fs = read_xwav(wav_path)
    if fs != FEATURE_FS:
        if fs % FEATURE_FS:
            raise ValueError(f"Cannot decimate {fs} Hz to {FEATURE_FS} Hz")
        t0 = time.time()
        x = decimate(x, fs // FEATURE_FS)
        if verbose:
            print(f"decimated {fs} Hz -> {FEATURE_FS} Hz ({time.time() - t0:.0f} s)")
    coeffs = matlab_mfcc(
        x,
        FEATURE_FS,
        window_sec=FEATURE_WINDOW_SEC,
        hop_sec=FEATURE_HOP_SEC,
        chunk_frames=50_000,
        dtype=np.float32,
        verbose=verbose,
    )
    frame_rate = 1.0 / FEATURE_HOP_SEC
    _save_atomic(cache_path, coeffs=coeffs, frame_rate=frame_rate)
    return coeffs, frame_rate


def block_average(T, k):
    """Average non-overlapping blocks of ``k`` points (NaN-aware; all-NaN -> NaN)"""
    if k == 1:
        return np.asarray(T, dtype=np.float64)
    n = len(T) // k
    blocks = np.asarray(T[: n * k], dtype=np.float64).reshape(n, k)
    finite = np.isfinite(blocks)
    counts = finite.sum(axis=1)
    sums = np.where(finite, blocks, 0.0).sum(axis=1)
    out = np.full(n, np.nan)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def find_offset(short, long, fs, expected_sec, probe_at_sec=60, probe_sec=20, search_sec=180):
    """
    Locate ``short[probe_at : probe_at + probe]`` inside ``long`` near
    ``expected_sec + probe_at_sec`` with a normalized cross-correlation. Returns
    ``(offset_sec, peak_correlation)`` where ``offset_sec`` is where ``short`` starts.
    """
    from scipy.signal import correlate

    p0 = int(probe_at_sec * fs)
    probe = np.asarray(short[p0 : p0 + int(probe_sec * fs)], dtype=np.float64)
    probe = (probe - probe.mean()) / probe.std()
    c = int((expected_sec + probe_at_sec) * fs)
    a = max(0, c - int(search_sec * fs))
    region = np.asarray(long[a : c + int((search_sec + probe_sec) * fs)], dtype=np.float64)
    L = len(probe)
    xc = correlate(region, probe, mode="valid", method="fft")
    cs = np.concatenate([[0.0], np.cumsum(region)])
    cs2 = np.concatenate([[0.0], np.cumsum(region**2)])
    mean = (cs[L:] - cs[:-L]) / L
    var = np.maximum((cs2[L:] - cs2[:-L]) / L - mean**2, 0.0)
    norm = np.sqrt(var) * L
    r = xc / np.where(norm > 0, norm, np.inf)
    k = int(np.argmax(r))
    return (a + k) / fs - probe_at_sec, float(r[k])


def _save_atomic(path, **arrays):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def compute_pan_mp(
    T, min_m, max_m, step, cache_path=None, excl_denom=1, save_every=5, verbose=True
):
    """
    Exact pan matrix profile via ``stumpy.stimp(percentage=1.0)``, computed in
    breadth-first window size order. Progress is saved to ``cache_path`` every
    ``save_every`` window sizes and resumed from there on the next call.
    """
    config.STUMPY_EXCL_ZONE_DENOM = excl_denom
    pmp = stumpy.stimp(T, min_m=min_m, max_m=max_m, step=step, percentage=1.0)
    n_windows = len(pmp.M_)

    if cache_path is not None and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if (
            np.array_equal(cached["T"], T)
            and np.array_equal(cached["M"], pmp.M_)
            and int(cached["excl_denom"]) == excl_denom
        ):
            pmp._PAN[:] = cached["PAN"]
            pmp._n_processed = int(cached["n_processed"])
            if verbose:
                print(f"Resumed {pmp._n_processed}/{n_windows} window sizes from cache")

    def save():
        if cache_path is not None:
            _save_atomic(
                cache_path,
                T=T,
                M=pmp.M_,
                PAN=pmp._PAN,
                n_processed=pmp._n_processed,
                excl_denom=excl_denom,
            )

    t0 = time.time()
    start = pmp._n_processed
    while pmp._n_processed < n_windows:
        pmp.update()
        done = pmp._n_processed
        if done % save_every == 0 or done == n_windows:
            save()
        if verbose:
            elapsed = time.time() - t0
            eta = elapsed / (done - start) * (n_windows - done)
            print(
                f"\r{done}/{n_windows} window sizes ({elapsed:.0f} s, ~{eta:.0f} s left)",
                end="",
                flush=True,
            )
    if verbose and start < n_windows:
        print()
    return pmp


def motif_strength(D, ms, score="pearson", max_m=None):
    """
    Convert raw z-normalized Euclidean distances ``D`` (one row per window size in
    ``ms``) to a motif strength where larger is better.

    * ``"pearson"``: the Pearson correlation ``1 - d^2 / (2m)``
    * ``"weighted"``: ``sqrt(m / max_m) * (1 - d / (2 sqrt(m)))``, the distance margin
      below its maximum ``2 sqrt(m)``, which rewards longer motifs
    """
    ms = np.asarray(ms, dtype=np.float64)[:, None]
    if max_m is None:
        max_m = ms.max()
    dn = np.minimum(D / (2.0 * np.sqrt(ms)), 1.0)
    if score == "pearson":
        S = 1.0 - 2.0 * dn**2
    elif score == "weighted":
        S = np.sqrt(ms / max_m) * (1.0 - dn)
    else:
        raise ValueError(f"Unknown score {score!r}")
    S[~np.isfinite(D)] = -np.inf
    return S


def nearest_neighbor(T, i, m, excl_denom=1):
    """Nearest non-trivial neighbor ``(j, distance)`` of subsequence ``T[i : i + m]``"""
    D = stumpy.mass(T[i : i + m], T)
    excl = int(np.ceil(m / excl_denom))
    D[max(0, i - excl) : i + excl + 1] = np.inf
    j = int(np.argmin(D))
    return j, float(D[j])


def top_pan_motifs(S, ms, T, k=3, excl_denom=1):
    """
    Repeatedly take the strongest ``(m, i)`` in the strength surface ``S``, find the
    nearest neighbor ``j`` of ``T[i : i + m]``, and mask every subsequence (of every
    length) that overlaps either occurrence before searching for the next motif.
    """
    S = S.copy()
    motifs = []
    for _ in range(k):
        r, i = np.unravel_index(np.argmax(S), S.shape)
        m = int(ms[r])
        j, d = nearest_neighbor(T, int(i), m, excl_denom)
        motifs.append({"m": m, "i": int(i), "j": j, "d": d})
        for r2, m2 in enumerate(ms):
            for start in (int(i), j):
                S[r2, max(0, start - m2 + 1) : start + m] = -np.inf
    return motifs


def refine_motifs(
    motifs,
    T,
    delta=200,
    fine_step=10,
    window=1000,
    score="pearson",
    max_m=None,
    excl_denom=1,
    min_m=None,
    cache_path=None,
    verbose=True,
):
    """
    For every motif, compute exact matrix profiles for all lengths within ``+/- delta``
    of its length in steps of ``fine_step`` (limited to ``[min_m, max_m]``) and keep the
    strongest subsequence starting within ``+/- window`` of the original occurrence.
    Matrix profiles are cached by length in ``cache_path``.
    """
    config.STUMPY_EXCL_ZONE_DENOM = excl_denom
    rows = {}
    if cache_path is not None and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if np.array_equal(cached["T"], T) and int(cached["excl_denom"]) == excl_denom:
            rows = {int(m): P for m, P in zip(cached["ms"], cached["P"])}

    refined = []
    for mo in motifs:
        lengths = np.arange(mo["m"] - delta, mo["m"] + delta + 1, fine_step)
        lengths = lengths[
            (lengths >= (min_m if min_m is not None else 3))
            & (lengths <= (max_m if max_m is not None else len(T)))
        ]
        for m in lengths:
            if int(m) in rows:
                continue
            t0 = time.time()
            rows[int(m)] = stumpy.stump(T, int(m))[:, 0].astype(np.float64)
            if verbose:
                print(f"m={m}: {time.time() - t0:.1f} s", flush=True)
            if cache_path is not None:
                ms_sorted = sorted(rows)
                n = len(T) - min(ms_sorted) + 1
                P = np.full((len(ms_sorted), n), np.inf)
                for r, mm in enumerate(ms_sorted):
                    P[r, : len(rows[mm])] = rows[mm]
                _save_atomic(
                    cache_path, T=T, ms=np.array(ms_sorted), P=P, excl_denom=excl_denom
                )

        lo = max(0, mo["i"] - window)
        hi = mo["i"] + window + 1
        best = None
        for m in lengths:
            P = rows[int(m)][lo:hi]
            s = motif_strength(P[None, :], [m], score=score, max_m=max_m)[0]
            c = int(np.argmax(s))
            if best is None or s[c] > best[0]:
                best = (s[c], int(m), lo + c)
        _, m, i = best
        j, d = nearest_neighbor(T, i, m, excl_denom)
        refined.append({"m": m, "i": i, "j": j, "d": d})
    return refined


def pair_length_profile(T, a0, b0, span, ms, cache_path=None, verbose=True):
    """
    For every length ``m`` in ``ms``, find the best matching pair of subsequences with
    one starting in ``[a0, a0 + span)`` and the other in ``[b0, b0 + span)`` (an AB-join
    of the two regions). The regions must not overlap.

    Returns a structured array with fields ``m``, ``i``, ``j`` and ``d``.
    """
    dtype = [("m", np.int64), ("i", np.int64), ("j", np.int64), ("d", np.float64)]
    key = np.array([a0, b0, span])
    if cache_path is not None and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if (
            np.array_equal(cached["T"], T)
            and np.array_equal(cached["key"], key)
            and np.array_equal(cached["profile"]["m"], ms)
        ):
            return cached["profile"]

    profile = np.empty(len(ms), dtype=dtype)
    t0 = time.time()
    for r, m in enumerate(ms):
        m = int(m)
        mp = stumpy.stump(
            T[a0 : a0 + span + m - 1], m, T[b0 : b0 + span + m - 1], ignore_trivial=False
        )
        i = int(np.argmin(mp[:, 0]))
        profile[r] = (m, a0 + i, b0 + int(mp[i, 1]), float(mp[i, 0]))
        if verbose:
            print(f"\r{r + 1}/{len(ms)} lengths ({time.time() - t0:.0f} s)", end="")
    if verbose:
        print()
    if cache_path is not None:
        _save_atomic(cache_path, T=T, key=key, profile=profile)
    return profile


def nn_correlation(T, m):
    """
    Matrix profile of ``T`` with a non-overlapping exclusion zone (``m``), returned as
    the Pearson correlation ``r = 1 - d^2 / (2m)`` of every window with its nearest
    neighbor (NaN where undefined) together with the nearest neighbor indices.
    """
    config.STUMPY_EXCL_ZONE_DENOM = 1
    mp = stumpy.stump(T, int(m))
    d = mp[:, 0].astype(np.float64)
    r = np.full(len(d), np.nan)
    finite = np.isfinite(d)
    r[finite] = 1.0 - d[finite] ** 2 / (2.0 * m)
    return r, mp[:, 1].astype(np.int64)


def block_shuffle(T, block, rng):
    """Cut ``T`` into consecutive blocks of ``block`` points and shuffle their order"""
    starts = np.arange(0, len(T), block)
    order = rng.permutation(len(starts))
    return np.concatenate([T[s : s + block] for s in starts[order]])


def phase_surrogate(T, rng):
    """Fourier surrogate: same power spectrum (and autocorrelation) as ``T``, random phases"""
    x = np.asarray(T, dtype=np.float64)
    f = np.fft.rfft(x - x.mean())
    phases = rng.uniform(0, 2 * np.pi, len(f))
    phases[0] = 0.0
    if len(x) % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(np.abs(f) * np.exp(1j * phases), n=len(x)) + x.mean()


def length_scan(segments, lengths, rate, null="phase", null_block_frac=0.25, n_null=1,
                top_frac=0.1, seed=0, verbose=True):
    """
    For each motif length (in points) compute, pooled over all ``segments``:

    * ``best_r``, ``top_r`` (mean of the top ``top_frac`` of windows) and ``median_r``
      of the nearest neighbor correlation
    * the same statistics for null copies of each segment: ``null="phase"`` uses Fourier
      surrogates (same spectrum, no repeated waveforms); ``null="block"`` shuffles blocks
      of ``null_block_frac * m`` points (note: the block joins create artificial steps)
    * ``excess = top_r - null_top_r``

    Returns ``(DataFrame, offsets)`` where ``offsets[m]`` holds the nearest neighbor
    offsets (seconds) of the top ``top_frac`` windows.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows, offsets = [], {}
    t0 = time.time()
    for k, m in enumerate(lengths):
        m = int(m)
        real, null_r, offs = [], [], []
        for seg in segments:
            if len(seg) < 3 * m:
                continue
            r, I = nn_correlation(seg, m)
            ok = np.isfinite(r)
            real.append(r[ok])
            offs.append((np.abs(I - np.arange(len(I)))[ok], r[ok]))
            for _ in range(n_null):
                if null == "phase":
                    surrogate = phase_surrogate(seg, rng)
                else:
                    surrogate = block_shuffle(seg, max(1, int(null_block_frac * m)), rng)
                r0, _ = nn_correlation(surrogate, m)
                null_r.append(r0[np.isfinite(r0)])
        if not real:
            continue
        R, N = np.concatenate(real), np.concatenate(null_r)
        q, qn = np.quantile(R, 1 - top_frac), np.quantile(N, 1 - top_frac)
        offsets[m] = np.concatenate([o[rr >= q] for o, rr in offs]) / rate
        rows.append(
            {
                "length_s": m / rate,
                "m": m,
                "best_r": R.max(),
                "top_r": R[R >= q].mean(),
                "median_r": np.median(R),
                "null_best_r": N.max(),
                "null_top_r": N[N >= qn].mean(),
                "null_median_r": np.median(N),
            }
        )
        rows[-1]["excess"] = rows[-1]["top_r"] - rows[-1]["null_top_r"]
        if verbose:
            print(f"\r{k + 1}/{len(lengths)} lengths ({time.time() - t0:.0f} s)", end="", flush=True)
    if verbose:
        print()
    return pd.DataFrame(rows), offsets


def detrended_residual(lengths_s, y, deg=2):
    """Residual of ``y`` after a polynomial fit in ``log(length)``"""
    x = np.log(np.asarray(lengths_s, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    return y - np.polyval(np.polyfit(x, y, deg), x)


def local_maxima(y):
    """Indices of interior local maxima of ``y`` with a positive value"""
    y = np.asarray(y)
    return [i for i in range(1, len(y) - 1) if y[i] > 0 and y[i] >= y[i - 1] and y[i] >= y[i + 1]]


def autocorrelation(T, max_lag):
    """Autocorrelation of the z-normalized series (NaN treated as 0) for lags ``0..max_lag``"""
    x = np.asarray(T, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nanmean(x))
    x = (x - x.mean()) / x.std()
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[: max_lag + 1] / n
    return acf


def period_peaks(lags_s, values, n_peaks=5, prominence=0.01, min_lag_s=0.0):
    """The ``n_peaks`` most prominent peaks of ``values`` (e.g. an ACF or histogram)"""
    from scipy.signal import find_peaks

    lags_s = np.asarray(lags_s)
    idx, props = find_peaks(values, prominence=prominence)
    keep = lags_s[idx] >= min_lag_s
    idx, prom = idx[keep], props["prominences"][keep]
    order = np.argsort(prom)[::-1][:n_peaks]
    return [(float(lags_s[idx[o]]), float(prom[o])) for o in order]


def offset_histogram(offsets, lo_s, hi_s, n_bins=60):
    """Histogram of nearest neighbor offsets on log-spaced bins. Returns centers, density"""
    edges = np.geomspace(lo_s, hi_s, n_bins + 1)
    counts = np.zeros(n_bins)
    for o in offsets.values():
        counts += np.histogram(o, bins=edges)[0]
    density = counts / max(counts.sum(), 1) / np.diff(np.log(edges))
    return np.sqrt(edges[:-1] * edges[1:]), density


def motif_copies(T, seed, m, max_gap, start_thr=0.5, min_thr=0.0, step=0.01,
                 max_overlap=0.1, max_copies=80):
    """
    Copies of ``T[seed : seed + m]`` as in the experiment PDF: windows ranked by their
    correlation with the seed are accepted greedily if they overlap previously accepted
    copies by at most ``max_overlap``. The correlation threshold starts at ``start_thr``
    and drops by ``step`` until no stretch longer than ``max_gap`` points (including the
    start and end of ``T``) is left without a copy.
    """
    D = stumpy.mass(T[seed : seed + m], T)
    r = 1.0 - D**2 / (2.0 * m)
    r[~np.isfinite(r)] = -np.inf
    min_sep = max(1, int(round((1 - max_overlap) * m)))
    blocked = np.zeros(len(r), dtype=bool)
    accepted = []
    for idx in np.argsort(r)[::-1]:
        if r[idx] < min_thr or len(accepted) >= max_copies:
            break
        if not blocked[idx]:
            accepted.append(int(idx))
            blocked[max(0, idx - min_sep + 1) : idx + min_sep] = True
    accepted = np.array(accepted)
    acc_r = r[accepted]

    n = len(T)
    thr = start_thr
    while True:
        pos = np.sort(accepted[acc_r >= thr])
        stretches = np.diff(np.concatenate([[0], pos, [n - m]])) if len(pos) else np.array([n])
        if stretches.max() <= max_gap or thr - step < min_thr:
            break
        thr -= step

    gaps = np.diff(pos)
    covered = np.zeros(n, dtype=bool)
    for p in pos:
        covered[p : p + m] = True
    return {
        "threshold": thr,
        "positions": pos,
        "n_copies": len(pos),
        "median_gap": float(np.median(gaps)) if len(gaps) else np.nan,
        "gap_cv": float(gaps.std() / gaps.mean()) if len(gaps) > 1 else np.nan,
        "max_stretch": int(stretches.max()),
        "coverage": float(covered.mean()),
    }


def song_regularity(T, lengths, rate, max_gap_sec=20 * 60, verbose=True):
    """``motif_copies`` statistics for the best pair at each length (points)"""
    import pandas as pd

    rows = []
    for m in lengths:
        m = int(m)
        r, _ = nn_correlation(T, m)
        seed = int(np.nanargmax(r))
        cp = motif_copies(T, seed, m, int(max_gap_sec * rate))
        rows.append(
            {
                "length_s": m / rate,
                "seed_min": seed / rate / 60,
                "seed_r": float(r[seed]),
                "threshold": cp["threshold"],
                "copies": cp["n_copies"],
                "median_gap_min": cp["median_gap"] / rate / 60,
                "gap_cv": cp["gap_cv"],
                "max_stretch_min": cp["max_stretch"] / rate / 60,
                "coverage": cp["coverage"],
            }
        )
        if verbose:
            print(f"\r{len(rows)}/{len(lengths)} lengths", end="", flush=True)
    if verbose:
        print()
    return pd.DataFrame(rows)


def smooth3(y):
    """3-point moving average (2-point at the ends)"""
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 3:
        return y.copy()
    out = np.convolve(y, np.ones(3) / 3, mode="same")
    out[0], out[-1] = (y[0] + y[1]) / 2, (y[-2] + y[-1]) / 2
    return out


def choose_range(scan, periods_s=(), prior=None, frac=0.5, min_excess=0.02):
    """
    Decision rule for one band: the contiguous lengths around the peak of the (3-point
    smoothed) excess correlation ``E(L)`` with ``E >= frac * max(E)``. The range is
    accepted if the peak excess is at least ``min_excess``, the peak is localized (not at
    the first or last length, and the range does not span the whole band, ``flat``) and a
    positive bump of the detrended ``top_r`` or ``best_r`` or a repetition period (within
    a factor 1.5) supports it.
    """
    L = scan["length_s"].to_numpy()
    E = smooth3(scan["excess"].to_numpy())
    p = int(np.argmax(E))
    out = {"peak_s": float(L[p]), "peak_excess": float(E[p]), "prior": prior}
    if E[p] < min_excess:
        return out | {
            "lo_s": None, "hi_s": None, "at_edge": None, "flat": None, "accepted": False,
            "support": [f"peak excess {E[p]:.3f} < {min_excess}"],
            "threshold_r": float(scan["null_top_r"].iloc[p]),
            "prior_overlap": None if prior is None else False,
        }
    lo = p
    while lo > 0 and E[lo - 1] >= frac * E[p]:
        lo -= 1
    hi = p
    while hi < len(E) - 1 and E[hi + 1] >= frac * E[p]:
        hi += 1
    support = []
    for col in ("top_r", "best_r"):
        res = detrended_residual(L, scan[col])
        bumps = [i for i in local_maxima(res) if lo - 1 <= i <= hi + 1]
        if bumps:
            support.append(f"{col} bump at {', '.join(f'{L[i]:g}' for i in bumps)} s")
    for period in periods_s:
        if L[lo] / 1.5 <= period <= L[hi] * 1.5:
            support.append(f"period {period:.3g} s")
    flat = lo == 0 and hi == len(E) - 1
    localized = 0 < p < len(E) - 1 and not flat
    if not localized:
        support = support + ["E peak at the band edge" if not flat else "E flat over the band"]
    out |= {
        "lo_s": float(L[lo]),
        "hi_s": float(L[hi]),
        "at_edge": bool(lo == 0 or hi == len(E) - 1),
        "flat": bool(flat),
        "localized": bool(localized),
        "accepted": bool(localized and support),
        "support": support,
        "threshold_r": float(scan["null_top_r"].iloc[p]),
    }
    if prior is not None:
        out["prior_overlap"] = bool(out["lo_s"] <= prior[1] and out["hi_s"] >= prior[0])
    return out


def bump_range(scan, min_excess=0.02, col="top_r"):
    """
    Fallback when ``E(L)`` has no localized peak: among the interior positive bumps of the
    detrended ``col``, take the one with the largest smoothed excess (at least
    ``min_excess``) and return ``(lo_s, hi_s, peak_s)``, the lengths around it where the
    residual stays positive. None if there is no such bump.
    """
    L = scan["length_s"].to_numpy()
    E = smooth3(scan["excess"].to_numpy())
    res = detrended_residual(L, scan[col])
    bumps = [i for i in local_maxima(res) if E[i] >= min_excess]
    if not bumps:
        return None
    b = max(bumps, key=lambda i: E[i])
    lo = b
    while lo > 0 and res[lo - 1] > 0:
        lo -= 1
    hi = b
    while hi < len(res) - 1 and res[hi + 1] > 0:
        hi += 1
    return float(L[lo]), float(L[hi]), float(L[b])


def regular_range(reg, period_s, max_cv=0.3, max_stretch_cycles=1.5, gap_tol=0.25):
    """
    Song-level regularity criterion: the lengths of ``song_regularity`` output whose
    copies recur with a median gap within ``gap_tol`` of ``period_s``, gap CV at most
    ``max_cv`` and no hole longer than ``max_stretch_cycles`` periods. Returns
    ``(lo_s, hi_s, mask)``; ``lo_s``/``hi_s`` are None if no length qualifies.
    """
    gap_s = reg["median_gap_min"].to_numpy() * 60
    mask = (
        (np.abs(gap_s - period_s) <= gap_tol * period_s)
        & (reg["gap_cv"].to_numpy() <= max_cv)
        & (reg["max_stretch_min"].to_numpy() * 60 <= max_stretch_cycles * period_s)
    )
    if not mask.any():
        return None, None, mask
    L = reg["length_s"].to_numpy()[mask]
    return float(L.min()), float(L.max()), mask


def _has_two_finite_windows(Tw, m):
    """True if ``Tw`` contains two non-overlapping NaN-free windows of length ``m``"""
    cs = np.concatenate([[0], np.cumsum(np.isfinite(Tw))])
    ok = np.flatnonzero(cs[m:] - cs[:-m] == m)
    return len(ok) > 0 and ok[-1] - ok[0] >= m


def best_motif_in_window(Tw, ms, thresholds, score="excess"):
    """
    Best repeated pair in the window ``Tw`` over the lengths ``ms`` (points), using
    ``stumpy.stump`` with a non-overlapping exclusion zone. Windows touching NaN (blanked
    detections) are ignored. ``thresholds[k]`` is the minimum correlation at ``ms[k]``.
    Candidates are ranked by ``score``: ``"excess"`` (correlation minus the threshold at
    that length) or ``"weighted"`` (length-weighted normalized distance). Returns a dict
    ``m, i, j, r, d, score, threshold`` (``i < j`` relative to ``Tw``) or None.
    """
    config.STUMPY_EXCL_ZONE_DENOM = 1
    max_m = max(ms)
    best = None
    for m, thr in zip(ms, thresholds):
        m = int(m)
        if len(Tw) < 2 * m or not _has_two_finite_windows(Tw, m):
            continue
        mp = stumpy.stump(Tw, m)
        d = mp[:, 0].astype(np.float64)
        if not np.isfinite(d).any():
            continue
        k = int(np.argmin(np.where(np.isfinite(d), d, np.inf)))
        r = 1.0 - d[k] ** 2 / (2.0 * m)
        if score == "excess":
            s = r - thr
        else:
            s = np.sqrt(m / max_m) * (1.0 - d[k] / (2.0 * np.sqrt(m)))
        if best is None or s > best["score"]:
            i, j = sorted((k, int(mp[k, 1])))
            best = {"m": m, "i": i, "j": j, "r": float(r), "d": float(d[k]),
                    "score": float(s), "threshold": float(thr)}
    return best


def forward_motif_search(T, rate, lo_s, hi_s, threshold, window_sec=None, n_lengths=6,
                         score="excess", level="", cache_path=None, save_every=25,
                         max_iter=None, skip_rejected=True, verbose=True):
    """
    Forward-only sliding-window motif search over ``T`` (sampled at ``rate`` Hz).

    The window ``[lb, lb + W)`` (``W = window_sec``, default 15 x ``hi_s``, capped at the
    series) is searched for the best repeated pair with ``n_lengths`` log-spaced lengths
    in ``[lo_s, hi_s]``. ``threshold`` is the minimum correlation, a number or a
    function of the length in seconds. If the best pair passes, both occurrences are
    blanked (NaN) and ``lb`` moves to the end of the earlier one; otherwise the candidate
    is logged as rejected and ``lb`` moves by the shortest length. The search stops when
    fewer than two shortest lengths remain.

    With ``score="excess"`` a window is accepted exactly when some pair in it reaches the
    threshold, so after a rejection a doubled window ``[lb, lb + 2W)`` is checked: if it
    has no such pair, every sliding window inside it is rejected too, and they are logged
    as one row (``skipped`` = number of window positions it stands for).

    Returns a DataFrame with one row per iteration (accepted and rejected). With
    ``cache_path`` (CSV) the finished result is cached and an interrupted run resumes from
    ``cache_path + ".partial.csv"``.
    """
    import pandas as pd

    if cache_path is not None and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    T = np.array(T, dtype=np.float64)
    n = len(T)
    ms = np.unique(np.round(np.geomspace(lo_s, hi_s, n_lengths) * rate).astype(int))
    ms = ms[ms >= 3]
    thr_fn = threshold if callable(threshold) else (lambda _L, t=float(threshold): t)
    thresholds = [float(thr_fn(m / rate)) for m in ms]
    W = int(min(n, round((window_sec or 15 * hi_s) * rate)))
    min_m = int(ms[0])

    rows, lb, it = [], 0, 0
    partial = None if cache_path is None else cache_path + ".partial.csv"
    if partial is not None and os.path.exists(partial):
        prev = pd.read_csv(partial)
        rows = prev.to_dict("records")
        for row in rows:
            if row["accepted"]:
                T[row["i"] : row["i"] + row["m"]] = np.nan
                T[row["j"] : row["j"] + row["m"]] = np.nan
        lb, it = int(rows[-1]["next_lb"]), int(rows[-1]["iteration"]) + 1

    def save(path):
        tmp = path + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, path)

    t0 = time.time()
    prev_rejected, no_skip_before = False, -1
    while lb + 2 * min_m <= n and (max_iter is None or it < max_iter):
        end = min(n, lb + W)
        skipped = 1
        best = None
        if (skip_rejected and prev_rejected and score == "excess" and end < n
                and lb > no_skip_before):
            ext = min(n, lb + 2 * W)
            best = best_motif_in_window(T[lb:ext], ms, thresholds, score)
            if best is None or best["r"] < best["threshold"]:
                # windows starting at lb, lb + min_m, ..., all inside [lb, ext)
                skipped = 1 + (ext - end) // min_m
                end = ext
            else:
                # a passing pair lies ahead; no skipping until lb has moved past it
                no_skip_before = lb + best["i"]
                best = best_motif_in_window(T[lb:end], ms, thresholds, score)
        else:
            best = best_motif_in_window(T[lb:end], ms, thresholds, score)
        row = {"level": level, "iteration": it, "lb_min": lb / rate / 60,
               "window_end_min": end / rate / 60, "skipped": skipped}
        if best is None:
            nxt = lb + skipped * min_m
            row |= {"accepted": False, "length_s": np.nan, "start1_min": np.nan,
                    "start2_min": np.nan, "correlation": np.nan, "score": np.nan,
                    "threshold": np.nan, "m": 0, "i": -1, "j": -1}
        else:
            i, j, m = lb + best["i"], lb + best["j"], best["m"]
            accepted = best["r"] >= best["threshold"]
            if accepted:
                T[i : i + m] = np.nan
                T[j : j + m] = np.nan
                nxt = i + m
            else:
                nxt = lb + skipped * min_m
            row |= {"accepted": bool(accepted), "length_s": m / rate,
                    "start1_min": i / rate / 60, "start2_min": j / rate / 60,
                    "correlation": best["r"], "score": best["score"],
                    "threshold": best["threshold"], "m": m, "i": i, "j": j}
        row["next_lb"] = nxt
        prev_rejected = not row["accepted"]
        rows.append(row)
        lb, it = nxt, it + 1
        if partial is not None and it % save_every == 0:
            save(partial)
        if verbose and it % 20 == 0:
            n_acc = sum(r["accepted"] for r in rows)
            print(f"\r{level}: {lb / n:6.1%} of the series, {it} iterations, {n_acc} accepted, "
                  f"{time.time() - t0:.0f} s", end="", flush=True)
    if verbose:
        n_acc = sum(r["accepted"] for r in rows)
        print(f"\r{level}: done, {it} iterations, {n_acc} accepted, {time.time() - t0:.0f} s"
              + " " * 20)

    df = pd.DataFrame(rows)
    if cache_path is not None:
        if lb + 2 * min_m > n:
            save(cache_path)
            if os.path.exists(partial):
                os.remove(partial)
        else:
            save(partial)
    return df


def null_threshold(scan, col="null_top_r"):
    """Correlation threshold as a function of the length (s), interpolated in log(length)"""
    x = np.log(scan["length_s"].to_numpy(dtype=np.float64))
    y = scan[col].to_numpy(dtype=np.float64)
    return lambda length_s: float(np.interp(np.log(length_s), x, y))


def forward_level(coeffs, frame_rate, ranges, feature, level, tag, n_lengths=5, verbose=True):
    """
    Run ``forward_motif_search`` for one feature (``"c0"``/``"c1"``) and level using the
    range, frame rate and per-length null threshold found in Phase 1 (``ranges`` is the
    content of ``whale_length_ranges.json``). Cached as
    ``whale_forward_<level>_<feature>_<tag>.csv`` next to this module.
    """
    import pandas as pd

    cfg = ranges[feature][level]
    col = ranges["meta"]["feature_columns"][feature]
    rate = cfg["rate_hz"]
    T = block_average(coeffs[:, col], int(round(frame_rate / rate)))
    scan = pd.read_csv(os.path.join(HERE, f"whale_scan_{level}_{feature}.csv"))
    df = forward_motif_search(
        T, rate, cfg["lo_s"], cfg["hi_s"], null_threshold(scan), n_lengths=n_lengths,
        level=level, verbose=verbose,
        cache_path=os.path.join(HERE, f"whale_forward_{level}_{feature}_{tag}.csv"),
    )
    return df.assign(feature=feature)


def level_threshold(segments, m, rate, n_null=1, seed=0):
    """Null top-decile correlation at length ``m`` (points) for ``segments`` at ``rate``"""
    scan, _ = length_scan(segments, [m], rate, n_null=n_null, seed=seed, verbose=False)
    return float(scan["null_top_r"].iloc[0])


def top_motifs_with_copies(T, m, k, threshold, mp_cache=None, verbose=True):
    """
    Top-``k`` motifs of length ``m`` with all their copies. The seed of each motif is the
    unclaimed window with the smallest matrix profile value; its copies are all windows
    whose correlation with the seed is at least ``threshold``, accepted greedily without
    overlapping each other or the copies of earlier motifs (claimed regions are blanked).
    A seed with fewer than two copies is dropped together with the windows within m/2 of
    it. The matrix profile (non-overlapping
    exclusion zone) is cached in ``mp_cache``. Returns a DataFrame with one row per copy:
    ``motif, copy, start, r`` (``start`` in points, motifs numbered from 1).
    """
    import pandas as pd

    T = np.asarray(T, dtype=np.float64)
    n_win = len(T) - m + 1
    if mp_cache is not None and os.path.exists(mp_cache):
        P = np.load(mp_cache)["P"]
    else:
        config.STUMPY_EXCL_ZONE_DENOM = 1
        t0 = time.time()
        P = stumpy.stump(T, m)[:, 0].astype(np.float64)
        if verbose:
            print(f"matrix profile of {len(T):,} points, m = {m}: {time.time() - t0:.0f} s")
        if mp_cache is not None:
            _save_atomic(mp_cache, P=P, m=m)
    P = np.where(np.isfinite(P), P, np.inf)
    Tm = T.copy()
    rows, motif = [], 0
    while motif < k and np.isfinite(P).any():
        seed = int(np.argmin(P))
        D = stumpy.mass(T[seed : seed + m], Tm)
        r = 1.0 - D**2 / (2.0 * m)
        r[~np.isfinite(r)] = -np.inf
        cand = np.flatnonzero(r >= threshold)
        cand = cand[np.argsort(r[cand])[::-1]]
        blocked = np.zeros(n_win, dtype=bool)
        starts = []
        for idx in cand:
            if not blocked[idx]:
                starts.append(int(idx))
                blocked[max(0, idx - m + 1) : idx + m] = True
        if len(starts) < 2:
            # neighbouring windows share the (claimed) matches of this seed
            P[max(0, seed - m // 2) : seed + m // 2 + 1] = np.inf
            continue
        motif += 1
        for c, s in enumerate(sorted(starts)):
            rows.append({"motif": motif, "copy": c + 1, "start": s, "r": float(r[s])})
            Tm[s : s + m] = np.nan
            P[max(0, s - m + 1) : s + m] = np.inf
        if verbose:
            print(f"motif {motif}: seed {seed}, {len(starts)} copies")
    return pd.DataFrame(rows)


def level_top_motifs(coeffs, frame_rate, level, length_s, rate, segments=None, k=10, col=2,
                     n_null=1, verbose=True):
    """
    Threshold and top-``k`` motifs with copies for one level of the hierarchy, on column
    ``col`` of ``coeffs`` block-averaged to ``rate``. ``segments=(count, seconds)`` picks
    the null segments spread over the series (None: the whole series). Cached as
    ``whale_top10_<level>.csv`` (one row per copy, with ``threshold``) and the matrix
    profile as ``whale_mp_<level>.npz``.
    """
    import pandas as pd

    path = os.path.join(HERE, f"whale_top10_{level}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    T = block_average(coeffs[:, col], int(round(frame_rate / rate)))
    m = int(round(length_s * rate))
    if segments is None:
        segs = [T]
    else:
        count, seg_sec = segments
        seg = int(seg_sec * rate)
        segs = [T[s : s + seg] for s in np.linspace(0, len(T) - seg, count).astype(int)]
    thr = level_threshold(segs, m, rate, n_null=n_null)
    if verbose:
        print(f"{level}: m = {m} points at {rate} Hz, threshold r = {thr:.3f}")
    tab = top_motifs_with_copies(T, m, k, thr, os.path.join(HERE, f"whale_mp_{level}.npz"), verbose)
    tab["start_s"] = tab["start"] / rate
    tab["length_s"] = m / rate
    tab["threshold"] = thr
    tab.insert(0, "level", level)
    tmp = path + ".tmp"
    tab.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return tab


def rolling_mean(T, n):
    """Centered moving average over ``n`` points (NaN-aware, shorter at the edges)"""
    x = np.asarray(T, dtype=np.float64)
    ok = np.isfinite(x)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    i = np.arange(len(x))
    lo = np.clip(i - n // 2, 0, len(x))
    hi = np.clip(i - n // 2 + n, 0, len(x))
    cnt = cn[hi] - cn[lo]
    return np.where(cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan)


def containment(upper, lower, min_overlap=0.9):
    """
    Pairs ``(upper_idx, lower_idx)`` (row positions) where at least ``min_overlap`` of a
    lower-level copy lies inside an upper-level copy. Both DataFrames need ``start_s`` and
    ``length_s`` columns (seconds). Returns a DataFrame ``upper, lower, rel_start, rel_end``
    with the lower copy's position relative to the upper copy (0 = start, 1 = end).
    """
    import pandas as pd

    us, ul = upper["start_s"].to_numpy(float), upper["length_s"].to_numpy(float)
    ls, ll = lower["start_s"].to_numpy(float), lower["length_s"].to_numpy(float)
    order = np.argsort(ls)
    ls_sorted = ls[order]
    max_ll = ll.max() if len(ll) else 0.0
    out = []
    for u in range(len(us)):
        a, b = us[u], us[u] + ul[u]
        lo = np.searchsorted(ls_sorted, a - max_ll)
        hi = np.searchsorted(ls_sorted, b)
        idx = order[lo:hi]
        if not len(idx):
            continue
        ov = np.minimum(b, ls[idx] + ll[idx]) - np.maximum(a, ls[idx])
        idx = idx[ov >= min_overlap * ll[idx]]
        for j in idx:
            out.append((u, int(j), (ls[j] - a) / (b - a), (ls[j] + ll[j] - a) / (b - a)))
    return pd.DataFrame(out, columns=["upper", "lower", "rel_start", "rel_end"])


def _hierarchy_observed(upper, lower, min_overlap):
    pairs = containment(upper, lower, min_overlap)
    n_up = len(upper)
    u_motif = upper["motif"].to_numpy()
    l_motif = lower["motif"].to_numpy()
    contains = np.zeros(n_up, dtype=bool)
    contains[pairs["upper"].to_numpy()] = True
    labels = [set() for _ in range(n_up)]
    for u, l in zip(pairs["upper"], pairs["lower"]):
        labels[u].add(int(l_motif[l]))
    jac = []
    for h in np.unique(u_motif):
        idx = np.flatnonzero(u_motif == h)
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                A, B = labels[idx[x]], labels[idx[y]]
                jac.append(len(A & B) / len(A | B) if A | B else 0.0)
    up_ids, lo_ids = np.unique(u_motif), np.unique(l_motif)
    frac = np.zeros((len(up_ids), len(lo_ids)))
    for a, h in enumerate(up_ids):
        idx = np.flatnonzero(u_motif == h)
        for b, l in enumerate(lo_ids):
            frac[a, b] = np.mean([l in labels[i] for i in idx])
    inside = np.zeros(len(lower), dtype=bool)
    inside[pairs["lower"].to_numpy()] = True
    return {
        "pairs": pairs,
        "contain_rate": float(contains.mean()) if n_up else np.nan,
        "inside_rate": float(inside.mean()) if len(lower) else np.nan,
        "consistency": float(np.mean(jac)) if jac else np.nan,
        "frac": frac,
        "upper_ids": up_ids,
        "lower_ids": lo_ids,
    }


def hierarchy_stats(upper, lower, duration_s, n_null=500, seed=0, min_overlap=0.9):
    """
    Nesting test of ``lower``-level copies inside ``upper``-level copies (DataFrames with
    ``motif, start_s, length_s``). Observed statistics: ``contain_rate`` (upper copies
    containing at least one lower copy), ``inside_rate`` (lower copies inside an upper
    copy), ``consistency`` (mean pairwise Jaccard similarity of the lower-motif label sets
    of copies of the same upper motif) and ``frac`` (upper motif x lower motif: fraction
    of copies containing it). Null: all lower copies shifted circularly by one random
    offset (``n_null`` times). Returns the observed dict with ``null`` statistics,
    one-sided p-values and the enrichment ``frac / null mean frac``.
    """
    rng = np.random.default_rng(seed)
    obs = _hierarchy_observed(upper, lower, min_overlap)
    null = {"contain_rate": [], "inside_rate": [], "consistency": []}
    null_frac = np.zeros_like(obs["frac"])
    for _ in range(n_null):
        shift = rng.uniform(0, duration_s)
        shifted = lower.assign(
            start_s=np.mod(lower["start_s"].to_numpy(float) + shift, duration_s))
        o = _hierarchy_observed(upper, shifted, min_overlap)
        for key in null:
            null[key].append(o[key])
        null_frac += o["frac"]
    null_frac /= n_null
    obs["null"] = {key: np.array(v) for key, v in null.items()}
    obs["p"] = {
        key: float((1 + np.sum(np.array(v) >= obs[key])) / (1 + n_null)) for key, v in null.items()
    }
    obs["null_frac"] = null_frac
    obs["enrichment"] = (obs["frac"] + 1e-3) / (null_frac + 1e-3)
    return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--coef", type=int, default=2, help="MATLAB column (1-based)")
    parser.add_argument("--min-sec", type=float, default=27.5)
    parser.add_argument("--max-sec", type=float, default=100.0)
    parser.add_argument("--step", type=int, default=50, help="window step in frames")
    parser.add_argument("--excl-denom", type=int, default=1)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    coeffs, frame_rate = load_or_compute_mfcc()
    T = coeffs[:, args.coef - 1].astype(np.float64)
    min_m = int(round(args.min_sec * frame_rate))
    max_m = int(round(args.max_sec * frame_rate))
    cache = args.cache or os.path.join(
        HERE, f"whale_pan_mp_matlab_step{args.step}_excl{args.excl_denom}.npz"
    )
    print(f"T: {len(T):,} points, m = {min_m}..{max_m} step {args.step}, cache {cache}")
    compute_pan_mp(T, min_m, max_m, args.step, cache, excl_denom=args.excl_denom)


if __name__ == "__main__":
    main()
