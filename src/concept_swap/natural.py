"""Natural images (Imagenette validation split): download, crops centred on the readout RF, rotations.

Local: downloads fast.ai's Imagenette-320 archive once (https://github.com/fastai/imagenette), extracts it with
tarfile's "data" filter (blocks path traversal; https://docs.python.org/3/library/tarfile.html#tarfile-extraction-filter),
and returns 224x224 crops whose pixel (111, 111) is the image centre, optionally after rotating the full image about
that centre (so the readout unit sees the same scene rotated). Global context: natural-image counterfactuals for E3.
Imagenette is an ImageNet subset; images are never redistributed by this repo (only file names).
"""

import hashlib
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from concept_swap.config import QUICK_SEED_OFFSET, Config
from concept_swap.log import get_logger
from concept_swap.stimuli import CENTER, SIZE

log = get_logger(__name__)
SHORT = 332  # resize the shorter side to 332 px so a 224 crop rotated by any angle stays inside the image (112*sqrt2 < 166)
N_VAL = 3925  # validation images in Imagenette v2 320px (https://www.tensorflow.org/datasets/catalog/imagenette)
IMAGENETTE_SHA256 = "569b4497c98db6dd29f335d1f109cf315fe127053cedf69010d047f0188e158c"  # archive used for the reported run


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MB chunks (https://docs.python.org/3/library/hashlib.html)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def halves(seed: int, paths: list) -> tuple[list, list]:
    """Seeded split of the validation images into a selection half and a confirmatory half."""
    perm = np.random.default_rng(seed).permutation(len(paths))  # fixed shuffle
    return [paths[i] for i in perm[::2]], [paths[i] for i in perm[1::2]]  # even positions / odd positions


def image_halves(cfg: Config, paths: list) -> tuple[list, list]:
    """Selection and confirmatory halves of the Imagenette validation images.

    Local: the real halves always use the non-quick selection seed. `--quick` keeps every 10th photo of the real
    selection half and splits those again with its own seed, so smoke tests never pass a confirmatory photo through an
    intervention. Global context: the one split shared by E3 and the post-hoc phases."""
    sel, test = halves(cfg.selection_seed - (QUICK_SEED_OFFSET if cfg.quick else 0), paths)  # the reported run's halves
    return halves(cfg.selection_seed, sel[::10]) if cfg.quick else (sel, test)  # ~200 selection photos for quick


def ensure_imagenette(cfg: Config) -> Path:
    """Download (atomically, checksum-verified) and extract Imagenette-320 into data/imagenette/; return val/.

    Local: the archive is written to a .part file and renamed only when complete; its SHA-256 must equal the value
    recorded for the reported run; extraction is redone unless exactly N_VAL validation images are present.
    Global context: the selection/confirmatory halves depend on the exact image list, so a partial download must fail.
    """
    root = cfg.data_dir / "imagenette"
    val = root / "imagenette2-320" / "val"
    if val.exists() and len(list(val.glob("*/*.JPEG"))) == N_VAL:
        return val
    root.mkdir(parents=True, exist_ok=True)
    tgz = root / "imagenette2-320.tgz"
    if not tgz.exists():
        log.info("downloading %s (~342 MB)", cfg.imagenette_url)
        part = tgz.with_suffix(".tgz.part")
        urllib.request.urlretrieve(cfg.imagenette_url, part)  # https://docs.python.org/3/library/urllib.request.html
        part.replace(tgz)  # atomic rename: an interrupted download never leaves a "complete-looking" archive
    if sha256(tgz) != IMAGENETTE_SHA256:
        raise RuntimeError(f"{tgz.name} checksum mismatch; delete it and rerun")
    with tarfile.open(tgz) as tf:
        tf.extractall(root, filter="data")  # safe extraction of an untrusted archive (Python 3.12+; backported to 3.8.17+)
    assert len(list(val.glob("*/*.JPEG"))) == N_VAL, "incomplete Imagenette extraction"
    return val


def list_images(val: Path) -> list[Path]:
    """All validation JPEGs, sorted for determinism (3925 files in Imagenette v2)."""
    return sorted(val.glob("*/*.JPEG"))


def crop(path: Path, angle: float = 0.0, square: bool = False) -> np.ndarray:
    """224x224 RGB crop in [0, 1] whose pixel (CENTER, CENTER) is the image centre, after rotating the resized image
    counter-clockwise by `angle` degrees about its centre (PIL Image.rotate, bilinear).

    square=True first cuts the central SHORT x SHORT square; PIL then rotates by 90/180/270 deg with an exact
    transpose instead of bilinear resampling (Image.rotate fast path, lines 2519-2520 of
    https://github.com/python-pillow/Pillow/blob/13.0.0/src/PIL/Image.py), so no interpolation blur differs from the
    unrotated base. The 224 crop is identical either way for angle 0.
    """
    im = Image.open(path).convert("RGB")  # some Imagenette files are greyscale or CMYK
    s = SHORT / min(im.size)  # scale so the shorter side becomes SHORT
    im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.BILINEAR)
    if square:
        l, t = (im.width - SHORT) // 2, (im.height - SHORT) // 2  # central square, same centre pixel as below
        im = im.crop((l, t, l + SHORT, t + SHORT))
    if angle:
        im = im.rotate(angle, resample=Image.Resampling.BILINEAR)  # about the image centre, same canvas size
    cx, cy = im.width // 2, im.height // 2  # image centre (integer pixel)
    box = (cx - CENTER, cy - CENTER, cx - CENTER + SIZE, cy - CENTER + SIZE)  # puts the centre at crop pixel (111, 111)
    return np.asarray(im.crop(box), dtype=np.float32) / 255.0  # HWC float in [0, 1]


def load(paths: list[Path], angle: float = 0.0) -> np.ndarray:
    """Stack crops of many images (N, 224, 224, 3)."""
    return np.stack([crop(p, angle) for p in paths])  # one centred crop per path
