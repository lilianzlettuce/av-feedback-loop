"""
Combines decoded image (from audio loop) with live camera frame

General function signature:
    fn(camera, decoded, feedback, state) -> mixed image

feedback = 0  camera only (no loop)
feedback = 1  loop only (room gradually replaces the image with its own resonances)

decay < 1 (applied in loop.step) slowly drains the loop's energy -> with no new input it fades rather than locking onto a fixed pattern
"""
import numpy as np


# Linear mix: out = (1 - a) * camera + a * decoded
def crossfade(camera, decoded, feedback, state=None):
    return (1 - feedback) * camera + feedback * decoded

# Per-pixel max: bright things persist and trail like phosphor
def lighten(camera, decoded, feedback, state=None):
    return np.maximum(camera * (1 - feedback), decoded * feedback)

# Camera feed shows through only where it changed since the last frame
# A still room overtaken by loop -> slowly dissolves. Movement repaints parts of the image
def motion_gated(camera, decoded, feedback, state):
    prev = state.get("prev_camera", camera)
    state["prev_camera"] = camera
    motion = np.clip(np.abs(camera - prev) * 8.0, 0, 1)
    w = np.clip((1 - feedback) + motion, 0, 1)
    return w * camera + (1 - w) * decoded

# Map names to functions
BLENDS = {"crossfade": crossfade, "lighten": lighten, "motion": motion_gated}
