"""Everything between the speaker and the microphone.

Two channels share one interface, so the loop code never changes:

    room.process(audio_out: np.ndarray) -> audio_in: np.ndarray  (same length)

SimulatedRoom  models the physics in code (works anywhere, reproducible)
PhysicalRoom   plays through a real speaker and records a real mic

Both are *stateful*: a reverb tail (and filter memory) lasts longer than one
frame, so whatever spills past the end of frame k is carried into frame k+1.

Because the received audio is delayed by an unknown amount (the acoustic path
and the sound card's latency), each frame starts with a known sync pulse, and
FrameAligner finds it with a matched filter (cross-correlation).
"""
import numpy as np
from scipy.signal import butter, chirp, correlate, fftconvolve, sosfilt, sosfilt_zi

FS = 48_000


# =============================================================== sync pulse
def sync_pulse(fs: int = FS, ms: float = 10.0, f0: float = 600.0, f1: float = 150.0):
    """Hann-windowed downward chirp. It sounds like a soft kick drum.

    A sweep is used instead of a pure tone because its autocorrelation has a
    single sharp peak. A pure sine correlates almost equally well one cycle
    early or late, and the speaker's bass roll-off shifts its phase, so
    frames would land several rows off.
    """
    n = int(fs * ms / 1000)
    t = np.arange(n) / fs
    sweep = chirp(t, f0=f0, t1=t[-1], f1=f1, method="logarithmic")
    return (0.9 * np.hanning(n) * sweep).astype(np.float32)


class FrameAligner:
    """Streaming receiver: push recorded audio in, get aligned payloads out.

    For every frame it:
      1. cross-correlates the buffer with the known pulse (matched filter)
      2. normalizes by local energy so loudness doesn't change the peak height
      3. if the peak passes `threshold`: payload starts right after the pulse.
         Otherwise it free-runs from the current position (frame slips/tears,
         like VHS tracking). Lower the threshold for a stable lock and raise it
         for more glitches.
      4. AGC: the pulse has a known amplitude, so xcorr_peak / |pulse|^2 is a
         least-squares estimate of channel gain. The payload is divided by it.

    This adds one frame of latency (it needs a full frame plus the next pulse buffered).
    """

    def __init__(self, pulse: np.ndarray, frame_len: int, threshold: float = 0.35,
                 agc: bool = True):
        self.pulse = pulse
        self.P = len(pulse)
        self.F = frame_len                       # pulse + payload
        self.threshold = threshold
        self.agc = agc
        self.energy = float(np.sqrt(np.sum(pulse ** 2)))
        self.buf = np.zeros(0, dtype=np.float32)
        self.gain = 1.0
        self.locked = 0
        self.slipped = 0

    def _find(self, x):
        xc = correlate(x, self.pulse, mode="valid")
        local = np.sqrt(np.convolve(x ** 2, np.ones(self.P), mode="valid")) + 1e-9
        score = xc / (local * self.energy)
        i = int(np.argmax(score))
        if score[i] < self.threshold:
            return None
        g = abs(xc[i]) / self.energy ** 2
        self.gain = 0.7 * self.gain + 0.3 * max(g, 1e-3)   # smoothed to avoid flicker
        return i

    def push(self, audio: np.ndarray) -> list[np.ndarray]:
        self.buf = np.concatenate([self.buf, audio.astype(np.float32)])
        out = []
        while len(self.buf) >= self.F + self.P:
            start = self._find(self.buf[: self.F])
            if start is None:
                start, self.slipped = 0, self.slipped + 1
            else:
                self.locked += 1
            payload = self.buf[start + self.P: start + self.F]
            out.append(payload / self.gain if self.agc else payload.copy())
            self.buf = self.buf[start + self.F:]
        return out


# =============================================================== room impulse responses
def synthetic_ir(fs: int = FS, rt60: float = 0.15, wet: float = 0.25, seed: int = 1):
    """Direct sound plus an exponentially decaying noise tail.

    rt60 : time (s) for the tail to fall by 60 dB  ->  envelope exp(-6.9 t / rt60)
    wet  : RMS of the tail relative to the direct sound
    Returned at unit energy, so the reverb doesn't change overall loudness.
    """
    n = max(1, int(rt60 * fs))
    t = np.arange(n) / fs
    tail = np.random.default_rng(seed).standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir = tail * wet / (np.sqrt(np.sum(tail ** 2)) + 1e-12)
    ir[0] = 1.0
    return (ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32)


def shoebox_ir(fs: int = FS, room_dim=(5.0, 4.0, 3.0), rt60: float = 0.4,
               speaker=(1.0, 2.0, 1.5), mic=(3.5, 2.0, 1.5)) -> np.ndarray:
    """Physically modeled impulse response of a rectangular room (pyroomacoustics).

    Uses the image-source method: every wall reflection is modeled as a mirrored
    copy of the speaker. Positions are in meters, measured from one corner.
    """
    import pyroomacoustics as pra

    absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(absorption),
                       max_order=max_order)
    room.add_source(list(speaker))
    room.add_microphone(list(mic))
    room.compute_rir()
    ir = np.asarray(room.rir[0][0], dtype=np.float32)
    return ir / (np.sqrt(np.sum(ir ** 2)) + 1e-12)


# =============================================================== simulated room
def peaking_sos(f0: float, gain_db: float, q: float, fs: int = FS) -> np.ndarray:
    """Peaking EQ biquad (RBJ Audio EQ Cookbook), used to model a room mode."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
    return np.concatenate([b / a[0], a / a[0]])[None, :]


class SimulatedRoom:
    """speaker -> air -> mic, modeled stage by stage.

    Stage                 Physical cause                     Visible effect
    -------------------   --------------------------------   -----------------------
    high-pass             small speaker can't move air       solid shapes -> outlines
    low-pass              mic / speaker treble roll-off      horizontal blur
    room modes (peaks)    standing waves between walls       some rows/tones bloom
    reverb (convolution)  reflections off surfaces           ghosts, smearing
    delay                 sound travels ~343 m/s             handled by sync
    clock drift           DAC and ADC clocks differ slightly image shears/rolls
    noise                 mic self-noise, room tone          grain, slow fade to grey

    The losses multiply on every pass. Each stage alone keeps ~0.9 correlation,
    but all of them together keep much less, so small changes to these defaults
    change the decay rate a lot.
    """

    def __init__(self, fs: int = FS, ir: np.ndarray | None = None,
                 highpass_hz: float = 40.0, lowpass_hz: float = 9000.0,
                 room_modes_hz=(120.0, 340.0, 910.0), mode_gain_db: float = 2.0,
                 delay_ms: float = 3.0, drift_ppm: float = 50.0,
                 noise_db: float = -40.0, seed: int = 0):
        self.fs = fs
        sections = [butter(2, highpass_hz, "highpass", fs=fs, output="sos"),
                    butter(4, lowpass_hz, "lowpass", fs=fs, output="sos")]
        sections += [peaking_sos(f, mode_gain_db, 8.0, fs) for f in room_modes_hz]
        self.sos = np.vstack(sections)
        self.zi = sosfilt_zi(self.sos) * 0.0          # filter memory, starts silent

        self.ir = synthetic_ir(fs) if ir is None else ir
        self.tail = np.zeros(0, dtype=np.float32)     # reverb spill-over (overlap-add)

        self.delay_line = np.zeros(int(fs * delay_ms / 1000), dtype=np.float32)
        self.ratio = 1.0 + drift_ppm * 1e-6
        self.read_pos = 0.0
        self.drift_buf = np.zeros(0, dtype=np.float32)

        self.noise_amp = 10 ** (noise_db / 20)
        self.rng = np.random.default_rng(seed)

    def _reverb(self, x):
        """Overlap-add convolution: add last call's tail and save this call's tail."""
        y = fftconvolve(x, self.ir)           # length len(x) + len(ir) - 1
        y[: len(self.tail)] += self.tail       # tail is len(ir) - 1, so it always fits
        self.tail = y[len(x):].astype(np.float32)
        return y[: len(x)].astype(np.float32)

    def _delay(self, x):
        y = np.concatenate([self.delay_line, x])
        self.delay_line = y[len(x):]
        return y[: len(x)]

    def _drift(self, x):
        """Read the stream at a slightly wrong rate (linear-interpolation resampling)."""
        self.drift_buf = np.concatenate([self.drift_buf, x])
        idx = self.read_pos + np.arange(len(x)) * self.ratio
        idx = idx[idx < len(self.drift_buf) - 1]
        y = np.interp(idx, np.arange(len(self.drift_buf)), self.drift_buf)
        if len(idx):
            consumed = int(idx[-1])
            self.read_pos = idx[-1] + self.ratio - consumed
            self.drift_buf = self.drift_buf[consumed:]
        return np.pad(y, (0, len(x) - len(y))).astype(np.float32)

    def process(self, audio_out: np.ndarray) -> np.ndarray:
        y, self.zi = sosfilt(self.sos, audio_out, zi=self.zi)
        y = self._reverb(y.astype(np.float32))
        y = self._delay(y)
        y = self._drift(y)
        y = y + self.noise_amp * self.rng.standard_normal(len(y)).astype(np.float32)
        return y.astype(np.float32)


# =============================================================== physical room
class PhysicalRoom:
    """Real speaker and mic, with the same interface as SimulatedRoom.

    First version: sounddevice.playrec plays and records each frame in one call.
    Simple, but the stream restarts every frame, which leaves small gaps. When
    that becomes a problem, switch to one continuous sd.Stream with a callback.
    """

    def __init__(self, fs: int = FS, device=None):
        import sounddevice as sd  # imported here so offline use doesn't need PortAudio

        self.sd, self.fs, self.device = sd, fs, device

    def process(self, audio_out: np.ndarray) -> np.ndarray:
        rec = self.sd.playrec(audio_out.astype(np.float32), samplerate=self.fs,
                              channels=1, device=self.device, blocking=True)
        return rec[:, 0].astype(np.float32)
