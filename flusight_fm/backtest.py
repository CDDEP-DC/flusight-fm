"""Rolling-origin backtests on past data versions.

For each past reference date, the pipeline runs exactly as it would have that
week. It uses the version of the target data available then (rebuilt from the
hub's git history), revision history only up to then, and its own earlier
backtest forecasts as the nowcast prior. Outputs are written to backtests/<name>/
in the same layout as forecasts/, so `flusight evaluate` scores either one.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from . import config as C
from . import data, evaluate, forecast
from .models import ModelPool

log = logging.getLogger(__name__)


def run(start: dt.date, end: dt.date, name: str, models, targets, every: int = 1, resume: bool = True) -> dict:
    """Run the backtest and return the evaluation tables. With resume, weeks already done are skipped."""
    root = C.BACKTEST_DIR / name
    keys = ["hosp"] + (["ed"] if "ed" in targets else [])
    vintages = {key: data.load_vintages(key) for key in keys}
    locations = data.load_locations()
    pool = ModelPool()
    refs = pd.date_range(data.reference_date_for(start), end, freq=f"{7 * every}D")
    for ref in (r.date() for r in refs):
        done = root / "model-output" / C.model_id() / f"{ref}-{C.model_id()}.csv"
        if resume and done.exists():
            log.info("Backtest %s: %s already done; skipping", name, ref)
            continue
        frames = {key: data.vintage_for(vintages[key], ref) for key in keys}
        if frames["hosp"] is None:
            log.warning("No hospital data version for %s; skipping", ref)
            continue
        frames = {k: v for k, v in frames.items() if v is not None}
        log.info("=== Backtest %s: reference date %s ===", name, ref)
        past = {key: {r: df for r, df in vintages[key].items() if r <= ref} for key in keys}
        result = forecast.run(ref, frames, locations, models, targets, past, prior_dir=root, pool=pool)
        for w in result.warnings:
            log.warning(w)
        forecast.save(result, root)
    return evaluate.evaluate(root, combinations=True)
