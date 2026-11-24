# av-feedback-loop

Image ↔ audio feedback loop.
A live image is encoded as sound, played into a room, re-recorded, decoded and mixed with the camera again. Each pass loses information, so the image slowly dissolves into the room's own resonant frequencies.

## Setup (once)

```bash
# install uv: https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS / Linux

# create .venv, install Python 3.13 + all deps + loopdsp, write uv.lock
uv sync
```

Commit `uv.lock` after the first `uv sync`, then can get identical versions with `uv sync`

**Notebook in VS Code:** install _Python_, _Jupyter_ and _Ruff_ extensions, open `notebooks/01_generational_loss.ipynb`, then **Select Kernel → Python Environments → .venv**. The first cell turns on `%autoreload 2` so edits to `src/loopdsp/*.py` show up without restarting the kernel.

## Defaults

Sample rate: 48 kHz, Frame rate: 10 fps
Per-generation compute time:

Conversion methods:

- Raster
  - 4800 samples/frame (480 sync + 4320 image) -> 64x48 image
- FM
- DCT/FFT
- Spectrogram (Griffin-Lim)
