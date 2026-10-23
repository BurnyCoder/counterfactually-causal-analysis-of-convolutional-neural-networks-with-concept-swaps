"""Run configuration loaded from `.env`.

Local: turns environment variables (see `.env.example`) into one frozen, typed `Config` object.
Global context: every phase (`phases/*.py`) receives the same `Config`, so a run is fully described by
its `.env` file, which is also logged at start-up for reproducibility.
"""

import os  # os.environ access: https://docs.python.org/3/library/os.html#os.environ
from dataclasses import asdict, dataclass, replace  # https://docs.python.org/3/library/dataclasses.html
from pathlib import Path  # path handling: https://docs.python.org/3/library/pathlib.html

from dotenv import load_dotenv  # reads .env into os.environ: https://github.com/theskumar/python-dotenv

ROOT = Path(__file__).resolve().parents[2]  # repo root = two levels above src/concept_swap/


def _ints(s: str) -> tuple[int, ...]:
    """Parse "1,2,3" into (1, 2, 3).

    Local: comma-separated env strings to an int tuple. Global context: used for list-valued settings.
    """
    return tuple(int(v) for v in s.split(",") if v.strip())  # skip empty items from trailing commas


@dataclass(frozen=True)
class Config:
    """All tunable settings of the pipeline (frozen so phases cannot mutate shared state).

    Local: plain typed fields with the defaults mirrored in `.env.example`.
    Global context: passed to every phase; `asdict(cfg)` is logged and stored with each result file.
    """

    device: str = "cuda"  # torch device string
    data_dir: Path = ROOT / "data"  # third-party data + caches
    results_dir: Path = ROOT / "results"  # JSON/NPZ outputs
    figures_dir: Path = ROOT / "figures"  # PNG/GIF figures
    log_dir: Path = ROOT / "logs"  # timestamped run logs
    lrn_mode: str = "tf"  # "tf" (TensorFlow-exact LRN) or "lucent" (library default)
    n_theta: int = 72  # orientations sampled over 360 deg
    radii: tuple[int, ...] = (30, 45, 60, 80)  # arc radii in pixels (pilot: where 3b and 4a match published tuning best)
    widths: tuple[int, ...] = (3, 5)  # stroke widths in pixels
    arc_span_deg: float = 120.0  # angular extent of each arc (< 180 deg)
    jitter_px: int = 2  # max random translation of a stimulus
    pilot_seed: int = 0  # pilot split (development only, discarded)
    selection_seed: int = 1  # selection split (fit stats, templates, nulls)
    confirm_seed: int = 2  # confirmatory split (all tests)
    stability_seeds: tuple[int, ...] = (3, 4)  # extra confirmatory splits for a stability check
    sources_per_base: int = 4  # minimal pairs per base stimulus
    min_dtheta: int = 30  # minimum |theta_base - theta_source| in degrees
    tol_deg: float = 10.0  # decoder success tolerance in degrees
    n_null_sets: int = 2000  # random neuron sets per null distribution
    n_boot: int = 2000  # bootstrap replicates
    batch_size: int = 64  # full-network forward batch (small: the 8 GB laptop GPU must also hold cached windows)
    imagenette_url: str = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"  # fast.ai mirror
    rotations: tuple[int, ...] = (-90, -45, 45, 90, 180)  # natural-image rotation angles in degrees
    quick: bool = False  # smoke-test mode: tiny grids and few null sets

    def with_quick(self) -> "Config":
        """Return a shrunken copy for the `--quick` smoke test.

        Local: fewer orientations/radii/null sets. Global context: lets `main.py --quick` exercise the
        full pipeline end-to-end in minutes without touching the confirmatory settings.
        """
        # Seeds are offset by 100 so smoke tests never see the confirmatory stimuli (no peeking before pre-registration).
        return replace(self, quick=True, n_theta=24, radii=self.radii[:2], widths=self.widths[:1],
                       n_null_sets=20, n_boot=50, sources_per_base=2, selection_seed=self.selection_seed + 100,
                       confirm_seed=self.confirm_seed + 100, stability_seeds=tuple(s + 100 for s in self.stability_seeds),
                       results_dir=self.results_dir / "quick", figures_dir=self.figures_dir / "quick")

    def as_log_dict(self) -> dict:
        """Config as a JSON-serialisable dict for logs and result files. Paths are stored relative to the repo root,
        so results never leak the local username or directory layout."""
        rel = lambda p: p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name
        return {k: rel(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def load_config(env_file: Path | None = None) -> Config:
    """Build a `Config` from `.env` (if present) and the process environment.

    Local: each field falls back to the dataclass default when its variable is unset.
    Global context: the single entry point used by `main.py`; tests construct `Config()` directly.
    """
    load_dotenv(env_file or ROOT / ".env", override=False)  # existing env vars win over the file
    env = os.environ.get  # short alias for readability below
    base = Config()  # defaults
    return Config(
        device=env("DEVICE", base.device),
        data_dir=ROOT / env("DATA_DIR", "data"),
        results_dir=ROOT / env("RESULTS_DIR", "results"),
        figures_dir=ROOT / env("FIGURES_DIR", "figures"),
        log_dir=ROOT / env("LOG_DIR", "logs"),
        lrn_mode=env("LRN_MODE", base.lrn_mode),
        n_theta=int(env("N_THETA", base.n_theta)),
        radii=_ints(env("RADII", ",".join(map(str, base.radii)))),
        widths=_ints(env("WIDTHS", ",".join(map(str, base.widths)))),
        arc_span_deg=float(env("ARC_SPAN_DEG", base.arc_span_deg)),
        jitter_px=int(env("JITTER_PX", base.jitter_px)),
        pilot_seed=int(env("PILOT_SEED", base.pilot_seed)),
        selection_seed=int(env("SELECTION_SEED", base.selection_seed)),
        confirm_seed=int(env("CONFIRM_SEED", base.confirm_seed)),
        stability_seeds=_ints(env("STABILITY_SEEDS", ",".join(map(str, base.stability_seeds)))),
        sources_per_base=int(env("SOURCES_PER_BASE", base.sources_per_base)),
        min_dtheta=int(env("MIN_DTHETA", base.min_dtheta)),
        tol_deg=float(env("TOL_DEG", base.tol_deg)),
        n_null_sets=int(env("N_NULL_SETS", base.n_null_sets)),
        n_boot=int(env("N_BOOT", base.n_boot)),
        batch_size=int(env("BATCH_SIZE", base.batch_size)),
        imagenette_url=env("IMAGENETTE_URL", base.imagenette_url),
        rotations=_ints(env("ROTATIONS", ",".join(map(str, base.rotations)))),
    )
