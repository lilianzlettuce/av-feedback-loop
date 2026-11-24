"""One generation of the feedback loop.

    step(state, live_frame) -> (audio_out, decoded_img, state)

        live_frame ─┐
                    ├─ blend ─> mixed ─ encode ─> [sync|payload] = audio_out ─> room
    decoded (k-1) ──┘                                                            │
        ▲                                                                        ▼
        └──────────── decode <── aligned payload <── FrameAligner <── audio_in ──┘

The channel (simulated or physical) is held in the state, so swapping a
real speaker and mic for the simulation needs no change here.
"""
from dataclasses import dataclass, field

import numpy as np

from . import blend as _blend
from . import channel as _channel
from . import codecs as _codecs


@dataclass
class LoopConfig:
    fs: int = 48_000
    fps: float = 10.0
    shape: tuple[int, int] = (48, 64)    # (H, W) -> 3072 px
    codec: str = "raster"                # key in codecs.CODECS
    blend: str = "crossfade"             # key in blend.BLENDS
    feedback: float = 0.85
    decay: float = 0.995                 # gain on the loop path each generation
    sync_ms: float = 10.0
    sync_threshold: float = 0.35

    @property
    def frame_len(self) -> int:          # samples per video frame
        return round(self.fs / self.fps)

    @property
    def payload_len(self) -> int:        # samples left for the image
        return self.frame_len - int(self.fs * self.sync_ms / 1000)


@dataclass
class LoopState:
    cfg: LoopConfig
    room: object                         # anything with .process(audio) -> audio
    aligner: _channel.FrameAligner
    pulse: np.ndarray
    decoded: np.ndarray | None = None    # latest image back from the room
    last_mixed: np.ndarray | None = None # latest image sent into the room
    generation: int = 0
    blend_state: dict = field(default_factory=dict)


def make_state(cfg: LoopConfig, room=None) -> LoopState:
    """Build a fresh loop. room=None -> SimulatedRoom with default settings."""
    pulse = _channel.sync_pulse(cfg.fs, cfg.sync_ms)
    return LoopState(
        cfg=cfg,
        room=room if room is not None else _channel.SimulatedRoom(cfg.fs),
        aligner=_channel.FrameAligner(pulse, cfg.frame_len, cfg.sync_threshold),
        pulse=pulse,
    )


def step(state: LoopState, live_frame: np.ndarray | None):
    """Run one generation. live_frame=None means no camera input (pure loop).

    Returns (audio_out, decoded_img, state). decoded_img is None for the first
    frame or two, while the aligner fills its buffer.
    """
    cfg = state.cfg
    encode, decode = _codecs.CODECS[cfg.codec]

    # combine inputs
    if state.decoded is None:
        mixed = live_frame if live_frame is not None else np.zeros(cfg.shape, np.float32)
    else:
        cam = live_frame if live_frame is not None else np.zeros(cfg.shape, np.float32)
        loop_img = state.decoded * cfg.decay
        mixed = _blend.BLENDS[cfg.blend](cam, loop_img, cfg.feedback, state.blend_state)
    mixed = np.clip(mixed, 0, 1).astype(np.float32)

    # image -> audio, with the sync pulse in front
    audio_out = np.concatenate([state.pulse, encode(mixed, cfg.payload_len)])

    # through the room
    audio_in = state.room.process(audio_out)

    # find frames, audio -> image
    for payload in state.aligner.push(audio_in):
        state.decoded = decode(payload, cfg.shape)

    state.generation += 1
    state.last_mixed = mixed
    return audio_out, state.decoded, state

"""
    Offline experiment: feed `image` as the camera for `camera_gens` generations (default: all of them), then turn the camera off.

    Returns (frames, audio): frames[g] is the image that went *into* the room at generation g, and audio is the concatenated output stream.
 """
def run_generations(cfg: LoopConfig, image: np.ndarray, n_gens: int,
                    camera_gens: int | None = None, room=None):
    camera_gens = n_gens if camera_gens is None else camera_gens
    state = make_state(cfg, room)
    frames, audio = [], []
    for g in range(n_gens):
        a, _, state = step(state, image if g < camera_gens else None)
        frames.append(state.last_mixed)
        audio.append(a)
    return frames, np.concatenate(audio)
