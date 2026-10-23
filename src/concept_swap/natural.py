"""Natural images (Imagenette validation split): download, crops centred on the readout RF, rotations.

Local: downloads fast.ai's Imagenette-320 archive once (https://github.com/fastai/imagenette), extracts it with
tarfile's "data" filter (blocks path traversal; https://docs.python.org/3/library/tarfile.html#tarfile-extraction-filter),
and returns 224x224 crops whose pixel (111, 111) is the image centre, optionally after rotating the full image about
that centre (so the readout unit sees the same scene rotated). Global context: natural-image counterfactuals for E3.
Imagenette is an ImageNet subset; images are never redistributed by this repo (only file names).
"""

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from concept_swap.config import Config
from concept_swap.log import get_logger
from concept_swap.stimuli import CENTER, SIZE

log = get_logger(__name__)
SHORT = 332  # resize the shorter side to 332 px so a 224 crop rotated by any angle stays inside the image (112*sqrt2 < 166)


def ensure_imagenette(cfg: Config) -> Path:
    """Download and extract Imagenette-320 into data/imagenette/ if needed; return the validation folder."""
    root = cfg.data_dir / "imagenette"
    val = root / "imagenette2-320" / "val"
    if val.exists():
        return val
    root.mkdir(parents=True, exist_ok=True)
    tgz = root / "imagenette2-320.tgz"
    if not tgz.exists():
        log.info("downloading %s (~342 MB)", cfg.imagenette_url)
        urllib.request.urlretrieve(cfg.imagenette_url, tgz)  # https://docs.python.org/3/library/urllib.request.html
    with tarfile.open(tgz) as tf:
        tf.extractall(root, filter="data")  # safe extraction of an untrusted archive (Python >= 3.12)
    return val


def list_images(val: Path) -> list[Path]:
    """All validation JPEGs, sorted for determinism (3925 files in Imagenette v2)."""
    return sorted(val.glob("*/*.JPEG"))


def crop(path: Path, angle: float = 0.0) -> np.ndarray:
    """224x224 RGB crop in [0, 1] whose pixel (CENTER, CENTER) is the image centre, after rotating the resized image
    counter-clockwise by `angle` degrees about its centre (PIL Image.rotate, bilinear)."""
    im = Image.open(path).convert("RGB")
    s = SHORT / min(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.BILINEAR)
    if angle:
        im = im.rotate(angle, resample=Image.Resampling.BILINEAR)  # about the image centre, same canvas size
    cx, cy = im.width // 2, im.height // 2
    box = (cx - CENTER, cy - CENTER, cx - CENTER + SIZE, cy - CENTER + SIZE)
    return np.asarray(im.crop(box), dtype=np.float32) / 255.0


def load(paths: list[Path], angle: float = 0.0) -> np.ndarray:
    """Stack crops of many images (N, 224, 224, 3)."""
    return np.stack([crop(p, angle) for p in paths])
