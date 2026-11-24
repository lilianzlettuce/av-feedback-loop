"""Image <-> audio codecs as pure functions (arrays in, arrays out)

-------- Conventions --------
img: float32 array, shape (H, W), grayscale luminance in [0, 1]
audio: float32 array, shape (n,), roughly in [-1, 1]

Every scheme has:
- encode(img, n, ...)       -> audio of n samples
- decode(audio, shape, ...) -> image of the given (H, W) shape

n: max number of samples (or other val) per img frame (fs / fps minus the sync pulse)
Each encoder stretches or packs the image to fill it so all schemes run at the
same frame rate and can be swapped freely.
"""
import numpy as np
from scipy.fft import dctn, idctn
from scipy.signal import hilbert, istft, medfilt, stft

FS = 48_000
HEADROOM = 0.8  # peak level, leaves room for reverb build-up without clipping


# ----- Helpers -----

# Linearly resample a 1-D signal to exactly n samples
def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == n:
        return x
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x).astype(np.float32)

# Clip an image to [0, 1] and convert to float32
def _clip01(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# ----- raster -----
# Direct rasterization ("rastrogram"): scan rows like a CRT beam and use luminance as the waveform amplitude
# ~1 sample per pixel, best spectral efficiency of analog schemes
# Speakers can't reproduce DC, so luminance is centered on 0 (black = -1, white = +1)
# In the room, bass roll-off acts as a high-pass filter -> solid shapes become outlines; echo -> ghost images

def raster_encode(img: np.ndarray, n: int) -> np.ndarray:
    line = img.astype(np.float32).ravel() * 2.0 - 1.0
    return HEADROOM * fit_length(line, n)

def raster_decode(audio: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    x = fit_length(audio, shape[0] * shape[1]) / HEADROOM
    return _clip01((x + 1.0) / 2.0).reshape(shape)


# ----- scanline FM -----
# Brightness -> pitch of one continuous oscillator (very fast analog SSTV).
# Encode: instantaneous frequency f[n] = f_black + b[n](f_white - f_black);
#         phase = 2*pi * cumsum(f) / fs keeps the phase continuous (no clicks).
# Decode: analytic signal (Hilbert) -> unwrapped phase -> its derivative is
#         the instantaneous frequency -> map back to brightness.
# Volume changes don't affect brightness; reverb smears the frequency track.

def fm_encode(img, n, fs=FS, f_black=400.0, f_white=4000.0):
    b = fit_length(img.ravel(), n)
    f = f_black + b * (f_white - f_black)
    return (HEADROOM * np.sin(2 * np.pi * np.cumsum(f) / fs)).astype(np.float32)


def fm_decode(audio, shape, fs=FS, f_black=400.0, f_white=4000.0):
    phase = np.unwrap(np.angle(hilbert(audio)))
    inst_f = np.diff(phase, prepend=phase[0]) * fs / (2 * np.pi)
    inst_f = medfilt(inst_f, 5)  # remove spikes where noise makes the phase jump
    b = (inst_f - f_black) / (f_white - f_black)
    return _clip01(fit_length(b, shape[0] * shape[1]).reshape(shape))


# ----- spectrogram -----
# Image as a spectrogram: each column is an STFT time frame and each row a
# frequency. Rows follow a musical scale (minor pentatonic) rather than linear
# bins, so the result sounds like chords. The image only gives magnitudes, so
# Griffin-Lim estimates a consistent phase by alternating STFT and iSTFT.
# Room modes boost certain rows, which then glow brighter every generation.

PENTATONIC = (0, 3, 5, 7, 10)


def scale_freqs(n_rows: int, base_hz: float = 110.0, scale=PENTATONIC) -> np.ndarray:
    """Frequencies climbing a repeating musical scale (lowest first)."""
    out, octave = [], 0
    while len(out) < n_rows:
        for step in scale:
            if len(out) == n_rows:
                break
            out.append(base_hz * 2 ** ((12 * octave + step) / 12))
        octave += 1
    return np.array(out)


def _spec_params(n, shape, fs):
    H, W = shape
    hop = max(1, n // W)
    nperseg = 4 * hop
    freqs = np.fft.rfftfreq(nperseg, 1 / fs)
    # image row 0 is the top -> highest pitch
    bins = np.searchsorted(freqs, scale_freqs(H)[::-1]).clip(0, len(freqs) - 1)
    return hop, nperseg, bins, len(freqs)


def spectrogram_encode(img, n, fs=FS, gl_iters=16, seed=0):
    H, W = img.shape
    hop, nperseg, bins, n_freqs = _spec_params(n, img.shape, fs)
    kw = {"fs": fs, "nperseg": nperseg, "noverlap": nperseg - hop}
    n_frames = stft(np.zeros(n), **kw)[2].shape[1]

    mag = np.zeros((n_freqs, n_frames), dtype=np.float32)
    cols = np.linspace(0, W - 1, n_frames).round().astype(int)
    for r, b in enumerate(bins):
        mag[b] = np.maximum(mag[b], img[r, cols])

    # Griffin-Lim: keep the target magnitude, take the phase from the last STFT of the reconstruction, repeat
    # Wach pass makes the phase more consistent
    rng = np.random.default_rng(seed)
    phase = np.exp(2j * np.pi * rng.random(mag.shape))
    for _ in range(gl_iters):
        x = fit_length(istft(mag * phase, **kw)[1], n)
        Z = stft(x, **kw)[2][:, :n_frames]
        phase = np.exp(1j * np.angle(Z))
    x = fit_length(istft(mag * phase, **kw)[1], n)
    return (HEADROOM * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def spectrogram_decode(audio, shape, fs=FS):
    H, W = shape
    hop, nperseg, bins, _ = _spec_params(len(audio), shape, fs)
    Z = np.abs(stft(audio, fs=fs, nperseg=nperseg, noverlap=nperseg - hop)[2])
    rows = Z[bins]
    img = rows[:, np.linspace(0, rows.shape[1] - 1, W).round().astype(int)]
    return _clip01(img / (np.percentile(img, 99) + 1e-9))  # auto-gain


# ----- Lowest-N DCT -----
# 2D DCT, keep only the `keep` lowest-frequency coefficients in zig-zag order, and hold each one as a constant level for n/keep samples
# Fewer coefficients means image dissolves into soft blobs. Noise lands in the DCT domain, so it appears as cosine ripples rather than speckle.

# Zig-zag order: upper left lowest spatial frequencies first (low-pass filter)
def zigzag(h: int, w: int) -> list[tuple[int, int]]:
    return sorted(((i, j) for i in range(h) for j in range(w)),
                  key=lambda p: (p[0] + p[1], p[0] if (p[0] + p[1]) % 2 else p[1]))


def dct_encode(img, n, keep=256, scale=0.05):
    C = dctn(img - 0.5, norm="ortho")
    coeffs = np.array([C[i, j] for i, j in zigzag(*img.shape)[:keep]], dtype=np.float32)
    x = np.repeat(coeffs * scale, n // keep)
    return np.clip(fit_length(x, n), -1, 1)


def dct_decode(audio, shape, keep=256, scale=0.05):
    spc = len(audio) // keep
    x = fit_length(audio, spc * keep)
    # average each hold period = matched filter for a rectangular pulse
    coeffs = x.reshape(keep, spc).mean(axis=1) / scale
    C = np.zeros(shape, dtype=np.float32)
    for (i, j), c in zip(zigzag(*shape)[:keep], coeffs):
        C[i, j] = c
    return _clip01(idctn(C, norm="ortho") + 0.5)


# Registry: maps name -> (encode, decode)
CODECS = {
    "raster": (raster_encode, raster_decode),
    "fm": (fm_encode, fm_decode),
    "spectrogram": (spectrogram_encode, spectrogram_decode),
    "dct": (dct_encode, dct_decode),
}


def test_pattern(shape: tuple[int, int] = (48, 64)) -> np.ndarray:
    """Synthetic input (circle, bar, gradient) that makes artifacts easy to see."""
    H, W = shape
    y, x = np.mgrid[0:H, 0:W]
    img = 0.25 * x / W
    img[(x - W * 0.35) ** 2 + (y - H * 0.5) ** 2 < (H * 0.28) ** 2] = 1.0
    img[:, int(W * 0.7):int(W * 0.75)] = 1.0
    img[int(H * 0.8):, int(W * 0.6):] = 0.6
    return img.astype(np.float32)
