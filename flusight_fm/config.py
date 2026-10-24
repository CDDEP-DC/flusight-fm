"""Settings for the OHTJHU-BM pipeline.

Any setting here can be overridden in `flusight.toml` at the repo root (keys are
case-insensitive), for example:

    ensemble_method = "mean"
    models = ["chronos2", "timesfm", "moirai2"]
    prior_weight = 0.5

See TUNING.md for which settings are worth tuning and how.
"""
from __future__ import annotations

import datetime as dt
import os
import tomllib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FORECAST_DIR = REPO / "forecasts"   # weekly outputs (tracked in git)
BACKTEST_DIR = REPO / "backtests"   # backtest outputs
CACHE_DIR = Path(os.environ.get("FLUSIGHT_CACHE", REPO / "cache"))  # downloads (not tracked)
MODELS_DIR = REPO / "models"        # fine-tuned checkpoints (not tracked)

# ------------------------------------------------------------------- identity
TEAM_ABBR = "OHTJHU"
MODEL_ABBR = "BM"
HUB_REPO = "cdcepi/FluSight-forecast-hub"
HUB_BRANCH = "main"
BASELINE_MODEL = "FluSight-baseline"
TARGET_PATHS = {
    "hosp": "target-data/target-hospital-admissions.csv",
    "ed": "target-data/target-ed-visits-prop.csv",
}
LOCATIONS_PATH = "auxiliary-data/locations.csv"

# --------------------------------------------------------------------- targets
HOSP_TARGET = "wk inc flu hosp"
RATE_TARGET = "wk flu hosp rate change"
ED_TARGET = "wk inc flu prop ed visits"
PEAK_WEEK_TARGET = "peak week inc flu hosp"
PEAK_INC_TARGET = "peak inc flu hosp"
TARGET_NAMES = {"hosp": HOSP_TARGET, "ed": ED_TARGET}
ALL_TARGETS = ["hosp", "rate_change", "ed", "peak"]

QUANTILES = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])
MEDIAN_IDX = int(np.flatnonzero(QUANTILES == 0.5)[0])
HORIZONS = [-1, 0, 1, 2, 3]
RATE_CHANGE_HORIZONS = [0, 1, 2, 3]
# (stable, large) thresholds per horizon, as count columns from locations.csv
RATE_CHANGE_COLS = {0: ("count_rate0p3", "count_rate1p7"), 1: ("count_rate0p5", "count_rate3"),
                    2: ("count_rate0p7", "count_rate4"), 3: ("count_rate1", "count_rate5")}
RATE_CHANGE_CATEGORIES = ["large_decrease", "decrease", "stable", "increase", "large_increase"]
MIN_STABLE_COUNT = 10       # count changes under 10 admissions are always "stable"
MAX_POP_FRACTION = 0.3      # hub rejects admission quantiles above 30% of population
ED_MAX = 0.25               # hub flags ED proportion quantiles above this as implausible

SEASON_START = dt.date(2026, 10, 10)  # first and last Saturdays eligible for the peak week
SEASON_END = dt.date(2027, 5, 29)

# ---------------------------------------------------------------------- models
MODELS = ["chronos2", "timesfm", "moirai2", "timestral"]
MIN_MODELS = 2              # need this many successful component models per target
ENSEMBLE_METHOD = "median"  # "median", "mean" or "weighted"
ENSEMBLE_WEIGHTS: dict = {} # model -> weight, used when ENSEMBLE_METHOD = "weighted"
SPREAD_SCALE = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}  # widen (>1) or narrow (<1) ensemble intervals per horizon
TRANSFORM = "none"          # "none", "sqrt" or "log1p", applied before forecasting
CONTEXT_START = None        # e.g. "2022-08-01" to drop early history
MAX_GAP_WEEKS = 3           # interpolate gaps up to this long; history before a longer gap is dropped
MAX_STALE_WEEKS = 2         # skip a location whose latest report is older than this
DEVICE = os.environ.get("FLUSIGHT_DEVICE", "auto")  # "auto" (CUDA if available, else CPU), "cpu", "cuda", "mps"

# --------------------------------------------------------------------- nowcast
NOWCAST = True              # revise the latest weeks for expected backfill before forecasting
NOWCAST_WEEKS = 2           # how many of the most recent reported weeks to revise
REVISION_START = "2024-11-01"  # earliest data version used to learn revision patterns
REVISION_SETTLE_WEEKS = 4   # a week counts as "final" once it is this many weeks old
REVISION_SHRINK = 5.0       # pseudo-count pulling each location's revision stats toward all locations
REVISION_MIN_SD = 0.01      # floor on revision s.d. (log scale)
NOWCAST_FALLBACK_SD = 0.05  # revision s.d. (log scale) when there is no revision history
PRIOR_WEIGHT = 1.0          # weight on our own earlier forecast for a week (0 disables)
LOG_OFFSET = {"hosp": 1.0, "ed": 1e-4}  # offset in log(value + offset) for each dataset

# ------------------------------------------------------------------------ peak
N_PEAK_SAMPLES = 2000
PEAK_RHO = 0.9              # week-to-week correlation of forecast errors in sampled trajectories
SEED = 20261010

# ----------------------------------------------------------------- fine-tuning
CHRONOS2_FINETUNED = str(MODELS_DIR / "chronos2-ft" / "finetuned-ckpt")


def model_id() -> str:
    return f"{TEAM_ABBR}-{MODEL_ABBR}"


def hub_raw(path: str, ref: str | None = None) -> str:
    return f"https://raw.githubusercontent.com/{HUB_REPO}/{ref or HUB_BRANCH}/{path}"


def load_overrides(path: Path) -> list[str]:
    """Apply settings from a TOML file; returns the names that were overridden."""
    if not path.exists():
        return []
    changed = []
    for key, value in tomllib.loads(path.read_text()).items():
        name = key.upper()
        if name not in globals() or name.startswith("_") or callable(globals()[name]):
            raise KeyError(f"unknown setting '{key}' in {path}")
        if isinstance(globals()[name], dict) and isinstance(value, dict):
            # TOML keys are strings; keep integer keys (e.g. horizons) as integers
            value = {int(k) if str(k).lstrip("-").isdigit() else k: v for k, v in value.items()}
        globals()[name] = value
        changed.append(name)
    return changed
