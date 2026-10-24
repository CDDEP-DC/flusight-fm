"""Backfill nowcast: estimate what the most recent, still-incomplete weeks will be revised to.

For each of the last NOWCAST_WEEKS reported weeks, two estimates are combined,
each a normal distribution on log(value + offset):

1. The latest report, shifted by how reports at that lag (0 = newest week) have
   historically been revised. The revision history is learned from past versions
   of the hub's target data: each version's report is compared with the value the
   newest version has once that week is REVISION_SETTLE_WEEKS old. Each location's
   mean and s.d. are shrunk toward the all-location values (REVISION_SHRINK).
2. Our own earlier forecast for that week (from last week's submission), if
   there is one. It is weighted by precision and scaled by PRIOR_WEIGHT.

The models are then run on the series with these weeks replaced by the nowcast
medians, and horizon -1 is submitted from the nowcast distribution.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)


def _spread(x: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    lo, hi = np.quantile(x, [0.05, 0.95])
    return float(np.std(np.clip(x, lo, hi), ddof=1))


def revision_errors(vintages: dict[dt.date, pd.DataFrame], as_of: dt.date, offset: float) -> pd.DataFrame:
    """log(settled value + offset) - log(first-reported value + offset), by location and lag."""
    start = pd.Timestamp(C.REVISION_START).date()
    avail = {r: df.dropna(subset=["value"]) for r, df in vintages.items() if start <= r <= as_of}
    if len(avail) < 3:
        return pd.DataFrame(columns=["location", "lag", "ref", "err"])
    truth = avail.pop(max(avail))
    settled = truth["date"].max() - pd.Timedelta(weeks=C.REVISION_SETTLE_WEEKS)
    truth = truth.drop_duplicates(["location", "date"], keep="last").set_index(["date", "location"])["value"]

    rows = []
    for ref, df in avail.items():
        last = df["date"].max()
        for lag in range(C.NOWCAST_WEEKS):
            d = last - pd.Timedelta(weeks=lag)
            if d > settled or d not in truth.index.get_level_values(0):
                continue
            rep = df[df["date"] == d].drop_duplicates("location", keep="last").set_index("location")["value"]
            j = pd.concat([rep, truth.loc[d]], axis=1, keys=["rep", "fin"], join="inner")
            err = np.log(j["fin"] + offset) - np.log(j["rep"] + offset)
            rows += [(loc, lag, ref, e) for loc, e in err.items()]
    return pd.DataFrame(rows, columns=["location", "lag", "ref", "err"])


def revision_stats(vintages, as_of, offset) -> dict:
    """{(location, lag): (mean, sd, n)}, plus ("*", lag) for all locations pooled."""
    errs = revision_errors(vintages, as_of, offset)
    stats = {}
    for lag, g in errs.groupby("lag"):
        pool_mean, pool_sd = float(g["err"].median()), _spread(g["err"].to_numpy())
        pool_sd = max(pool_sd if np.isfinite(pool_sd) else C.NOWCAST_FALLBACK_SD, C.REVISION_MIN_SD)
        stats["*", lag] = (pool_mean, pool_sd, len(g))
        for loc, gl in g.groupby("location"):
            n = len(gl)
            w = n / (n + C.REVISION_SHRINK)
            sd_loc = _spread(gl["err"].to_numpy())
            sd = np.sqrt(w * sd_loc**2 + (1 - w) * pool_sd**2) if np.isfinite(sd_loc) else pool_sd
            stats[loc, lag] = (w * float(gl["err"].median()) + (1 - w) * pool_mean, max(sd, C.REVISION_MIN_SD), n)
    if stats:
        log.info("Revision history: %s", {f"lag {lag}": f"median {m:+.3f}, sd {s:.3f}, n={n}"
                                          for (loc, lag), (m, s, n) in stats.items() if loc == "*"})
    else:
        log.info("No revision history available; nowcast uses the reports as they are")
    return stats


def prior_forecasts(prior_dir: Path | None, target: str, dates, offset: float) -> dict:
    """{(location, date): (mu, sd)} on the log scale, from earlier submissions' horizon-0 forecasts."""
    out = {}
    if prior_dir is None or C.PRIOR_WEIGHT <= 0:
        return out
    folder = Path(prior_dir) / "model-output" / C.model_id()
    for d in dates:
        f = folder / f"{pd.Timestamp(d).date()}-{C.model_id()}.csv"
        if not f.exists():
            continue
        sub = pd.read_csv(f, dtype={"location": str, "output_type_id": str})
        sub = sub[(sub["target"] == target) & (sub["horizon"] == 0) & (sub["output_type"] == "quantile")]
        wide = sub.assign(q=sub["output_type_id"].astype(float)).pivot_table(index="location", columns="q", values="value")
        if not {0.25, 0.5, 0.75} <= set(wide.columns):
            continue
        lq = np.log(wide[[0.25, 0.5, 0.75]].clip(lower=0) + offset)
        sd = ((lq[0.75] - lq[0.25]) / 1.349).clip(lower=1e-3)
        for loc in wide.index:
            out[loc, pd.Timestamp(d)] = (float(lq.at[loc, 0.5]), float(sd[loc]))
    return out


def nowcast(series: dict[str, pd.Series], stats: dict, prior: dict, offset: float, adjust: bool = True):
    """Returns (revised series, table of nowcasts for the last NOWCAST_WEEKS weeks)."""
    rows, revised = [], {}
    for loc, s in series.items():
        s2 = s.copy()
        for lag in range(min(C.NOWCAST_WEEKS, len(s))):
            d, rep = s.index[-1 - lag], float(s.iloc[-1 - lag])
            mean, sd, n = stats.get((loc, lag)) or stats.get(("*", lag)) or (0.0, C.NOWCAST_FALLBACK_SD, 0)
            mu = np.log(rep + offset) + (mean if adjust else 0.0)
            p = prior.get((loc, d)) if adjust else None
            if p is not None:
                pm, psd = p[0], p[1] / np.sqrt(C.PRIOR_WEIGHT)
                prec = 1 / sd**2 + 1 / psd**2
                mu, sd = (mu / sd**2 + pm / psd**2) / prec, np.sqrt(1 / prec)
            med = max(float(np.exp(mu) - offset), 0.0)
            if adjust:
                s2.loc[d] = med
            rows.append(dict(location=loc, date=d, lag=lag, reported=rep, median=med, mu=mu, sd=sd,
                             n_revisions=n, prior_median=float(np.exp(p[0]) - offset) if p else np.nan))
        revised[loc] = s2
    return revised, pd.DataFrame(rows)
