"""An illustrated version of a camera frame, made by an image model.

This is a presentation layer only. The model fills in the set-up, the background
and the interior of the bubble from what it learns of such images; none of
those pixels were recorded. The measured outline band is protected and kept
from the recording, and every illustrated frame is labelled as such wherever it
is shown. Nothing here feeds a measurement, a CSV or a JSON export.

The key is read from ``OPENAI_API_KEY`` on the server and is never returned to a
client or written to a log.
"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
from scipy import ndimage

from naviernet.utils.logging import get_logger

log = get_logger(__name__)

API_URL = "https://api.openai.com/v1/images/edits"
KEY_ENV = "OPENAI_API_KEY"
MODEL_ENV = "NAVIERNET_STUDIO_IMAGE_MODEL"
DEFAULT_MODEL = "gpt-image-1"
# The measured edge band that is kept from the recording, in pixels either side.
PROTECTED_PX = 6
TIMEOUT_S = 180.0
LABEL = "illustration, not measured"
# Generated illustrated frames committed to the repository, one folder per run.
BUNDLED_DIR = Path("apps/studio/illustrations")
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

PROMPT = (
    "A microscope photograph of a single vapour bubble growing in a 300 micrometre "
    "wide liquid-filled channel. Remove the heaters, electrodes and metal stripes of "
    "the chip: the background is a plain, even, neutral grey. Keep the bubble's dark "
    "rim and its soft internal shading, with no new features, no text and no colour."
)


class IllustrationUnavailable(Exception):
    """No key or no model: the illustrated mode cannot run here."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str


def settings_from_env() -> Settings:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise IllustrationUnavailable(
            f"the illustrated image needs {KEY_ENV} set in the environment of the studio service"
        )
    return Settings(api_key=key, model=os.environ.get(MODEL_ENV, DEFAULT_MODEL))


def bundled_path(repo_root: object, run_id: str, frame: int) -> Path | None:
    """The illustrated frame committed to the repository, if there is one.

    Bundled frames are generated once, offline, and need no key to be served.
    """
    if not SAFE_NAME.match(run_id):
        return None
    candidate = Path(str(repo_root)) / BUNDLED_DIR / run_id / f"frame_{frame:02d}.png"
    return candidate if candidate.is_file() else None


def protected_band(bubble: np.ndarray) -> np.ndarray:
    """The measured outline band, which is kept from the recording."""
    outer = ndimage.binary_dilation(bubble, iterations=PROTECTED_PX)
    inner = ndimage.binary_erosion(bubble, iterations=PROTECTED_PX)
    return outer & ~inner


def edit_mask_png(protected: np.ndarray) -> bytes:
    """An RGBA mask: transparent where the model may draw, opaque where it may not."""
    alpha = np.where(protected, 255, 0).astype(np.uint8)
    rgba = np.zeros((*protected.shape, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    return _png(rgba)


def composite(generated: np.ndarray, recorded: np.ndarray, protected: np.ndarray) -> np.ndarray:
    """The generated frame, with the protected band restored from the recording."""
    return np.where(protected, recorded, generated)


def request_illustration(image_png: bytes, mask_png: bytes, cfg: Settings) -> bytes:
    """Ask the image model to edit the frame; returns the PNG it made."""
    files = {
        "image": ("frame.png", image_png, "image/png"),
        "mask": ("mask.png", mask_png, "image/png"),
    }
    data = {"model": cfg.model, "prompt": PROMPT, "size": "1024x1024", "n": "1"}
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    with httpx.Client(timeout=TIMEOUT_S) as client:
        response = client.post(API_URL, headers=headers, data=data, files=files)
    if response.status_code != 200:
        log.warning("image model returned HTTP %s for an illustration", response.status_code)
        raise IllustrationUnavailable(f"the image model returned HTTP {response.status_code}")
    payload = response.json()
    encoded = payload["data"][0].get("b64_json")
    if not encoded:
        raise IllustrationUnavailable("the image model returned no image")
    return base64.b64decode(encoded)


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()
