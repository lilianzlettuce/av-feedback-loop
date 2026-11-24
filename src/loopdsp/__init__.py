""" DSP core for image <-> audio feedback loop

Modules
-------
codecs   pure functions: image -> audio (encode) and audio -> image (decode)
channel  what happens between speaker and mic, plus sync pulse that marks where each frame starts
blend    combine decoded image with live camera frame
loop     one full generation of the loop
"""

from . import blend, channel, codecs, loop
from .loop import LoopConfig, LoopState, step

__all__ = ["codecs", "channel", "blend", "loop", "LoopConfig", "LoopState", "step"]
