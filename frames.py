"""The one place camera frames get resized.

The recorder archives at RENDER_SIZE, load_data reduces shards to whatever the
training arena declares, and a rollout has to build its image the same way. If
those disagree the policy just gets quietly worse -- nothing raises, nothing
logs -- so they all come through here.

Only integer reductions are allowed, and that restriction is the point. PIL
BOX, cv2 INTER_AREA and torch interpolate(mode='area') all claim to average
over the source footprint, but agree only when the ratio divides evenly.
Worst-pixel difference against BOX, measured on this scene:

    2x, 4x, 5x        cv2  1    torch  1      (uint8 rounding)
    2.5x, 2.857x      cv2 47    torch 77

At an integer ratio every output pixel is exactly an NxN block, so there is
nothing to disagree about. A resize done anywhere -- on the GPU in a
training loop, by a pretrained model's own image processor -- still matches
this one, and correctness stops depending on everyone calling this function.

Render large and reduce here rather than rendering small. MuJoCo rasterises one
sample per pixel, so a small render decides each edge pixel in or out, while
reducing from a large one averages ~6 samples and reconstructs partial
coverage. The two differ by 1.5/255 overall but 7.3 on the edgiest tenth of the
image, which is most of what a conv net's first layer sees. Rendering at 896
costs 0.4 ms/step over 256, so matching is close to free.
"""

import numpy as np
from PIL import Image

# Even divisors of RENDER_SIZE, dropping those too small to be images. 224 is
# the VLM target, 128 the small-CNN one, 896 a pass-through. Changing the render
# size means recomputing the list, not just editing the number.
RENDER_SIZE = 896
SUPPORTED_SIZES = (64, 112, 128, 224, 448, 896)


def validate(width, size):
    """Raise unless width can be reduced to size. Call before committing to it."""
    if width < size:
        raise ValueError(
            f"frames are {width}px but {size}px was asked for. That would mean "
            f"upsampling; work at {width}px or smaller.")

    if size not in SUPPORTED_SIZES:
        raise ValueError(
            f"{size} is not a supported size. Must be one of "
            f"{list(SUPPORTED_SIZES)}.")

    if width % size:
        raise ValueError(
            f"{size} is a supported size but does not divide {width}px frames "
            f"evenly. SUPPORTED_SIZES describes the {RENDER_SIZE}px this project "
            f"renders at, so {width}px frames came from somewhere else -- an "
            f"image_size that was not updated, or shards from an older render.")


def resize(frames, size):
    """Reduce camera frames to size x size. Accepts (H,W,3) or (N,H,W,3).

    BOX averages over the source pixels each output pixel covers. Upsampling
    and fractional ratios both raise -- see the module docstring for why the
    ratio has to be integral.
    """
    single = frames.ndim == 3
    if single:
        frames = frames[None]

    width = frames.shape[1]
    if width == size:
        return frames[0] if single else frames

    validate(width, size)

    # PIL crawls on the flipped view MuJoCo hands back (negative row stride),
    # 2.8 ms against 0.8 for the same result. The copy costs 0.02 ms.
    frames = np.ascontiguousarray(frames)

    out = np.empty((len(frames), size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        out[i] = np.asarray(Image.fromarray(frame).resize((size, size), Image.BOX))
    return out[0] if single else out
