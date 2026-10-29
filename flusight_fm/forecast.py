"""One week's forecast: nowcast, component models, ensemble, and every submission target."""
from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config as C
from . import nowcast as nc
from .models import FREQ, ModelPool

log = logging.getLogger(__name__)

_TRANSFORMS = {
    "none": (lambda x: x, lambda x: x),
    "sqrt": (np.sqrt, lambda x: np.square(np.maximum(x, 0))),
    "log1p": (np.log1p, np.expm1),
}


def _z(levels) -> np.ndarray:
    return np.array([NormalDist().inv_cdf(float(p)) for p in levels])


@dataclasses.dataclass
class Result:
    ref_date: dt.date
    submission: pd.DataFrame
    problems: list[str]
    warnings: list[str]
    last_obs: dict           # key -> Timestamp of the last reported week
    reported: dict           # key -> {location: Series as reported}
    revised: dict            # key -> {location: Series after nowcast}
    nowcasts: dict           # key -> DataFrame from nowcast.nowcast
    components: dict         # key -> {model: {location: array(steps, 23)}}
    ensembles: dict          # key -> {location: array(steps, 23)}
    horizon_q: dict          # key -> {(location, horizon): array(23)}
    peaks: dict              # location -> {"weeks", "probs", "size"}


# ------------------------------------------------------------------------- data
def build_series(target: pd.DataFrame, locations: pd.DataFrame, ref_date: dt.date):
    """One gap-free weekly series per location, using only weeks before the reference date.

    Returns (series, last reported week, locations skipped).
    """
    cutoff = pd.Timestamp(ref_date) - pd.Timedelta(weeks=1)
    df = target.dropna(subset=["value"])
    df = df[(df["date"] <= cutoff) & df["location"].isin(locations["location"])]
    if C.CONTEXT_START:
        df = df[df["date"] >= pd.Timestamp(C.CONTEXT_START)]
    if (df["date"].dt.dayofweek != 5).any():
        raise ValueError("target data contains dates that are not Saturdays")
    last_obs = df["date"].max()

    series, skipped = {}, []
    for loc, g in df.groupby("location"):
        s = g.drop_duplicates("date", keep="last").set_index("date")["value"].astype(float).sort_index()
        if s.index.max() < last_obs - pd.Timedelta(weeks=C.MAX_STALE_WEEKS):
            skipped.append(loc)
            continue
        s = s.reindex(pd.date_range(s.index.min(), last_obs, freq=FREQ))
        s = s.interpolate(limit=C.MAX_GAP_WEEKS, limit_area="inside").ffill(limit=C.MAX_STALE_WEEKS)
        if s.isna().any():  # a gap too long to fill: keep only the history after it
            s = s[s.index > s[s.isna()].index.max()]
        series[loc] = s.clip(lower=0)
    return series, last_obs, sorted(set(locations["location"]) - set(series))


def weeks_between(start, end) -> int:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days // 7


def season_bounds(ref_date: dt.date) -> tuple[dt.date, dt.date]:
    """First and last Saturdays eligible for the peak week, for the season containing ref_date.

    The current season comes from SEASON_START/SEASON_END; other seasons (backtests)
    follow the same pattern: the Saturday on/after Oct 8 to the Saturday on/after May 25.
    """
    year = ref_date.year if ref_date.month >= 8 else ref_date.year - 1
    if year == C.SEASON_START.year:
        return C.SEASON_START, C.SEASON_END
    saturday = lambda d: d + dt.timedelta(days=(5 - d.weekday()) % 7)
    return saturday(dt.date(year, 10, 8)), saturday(dt.date(year + 1, 5, 25))


# ---------------------------------------------------------------------- models
def to_flusight_levels(levels, q: np.ndarray) -> np.ndarray:
    """Map quantiles at a model's native levels onto FluSight's 23 levels.

    Interpolates linearly in normal-score space and extrapolates the tails with
    the slope of the outermost pair, so models that only give 0.1-0.9 still get
    0.01/0.025/0.975/0.99.
    """
    levels = np.asarray(levels, float)
    order = np.argsort(levels)
    z, zt = _z(levels[order]), _z(C.QUANTILES)
    q = np.sort(np.asarray(q, float)[..., order], axis=-1)
    out = np.empty(q.shape[:-1] + (len(zt),))
    lo, hi = zt < z[0], zt > z[-1]
    for idx in np.ndindex(q.shape[:-1]):
        v = q[idx]
        r = np.interp(zt, z, v)
        r[lo] = v[0] + (zt[lo] - z[0]) * (v[1] - v[0]) / (z[1] - z[0])
        r[hi] = v[-1] + (zt[hi] - z[-1]) * (v[-1] - v[-2]) / (z[-1] - z[-2])
        out[idx] = r
    return out


def _predict(model, series, n_steps):
    fwd, inv = _TRANSFORMS[C.TRANSFORM]
    tseries = {loc: pd.Series(fwd(s.to_numpy()), index=s.index) for loc, s in series.items()}
    levels, preds = model.predict(tseries, n_steps)
    missing = set(series) - set(preds)
    if missing:
        raise RuntimeError(f"no forecast for locations {sorted(missing)}")
    out = {}
    for loc in series:
        q = np.asarray(preds[loc], float)
        if q.shape != (n_steps, len(levels)):
            raise RuntimeError(f"bad forecast for {loc}: shape {q.shape}, expected {(n_steps, len(levels))}")
        if not np.isfinite(q).all():
            raise RuntimeError(f"bad forecast for {loc}: {np.size(q) - np.isfinite(q).sum()} non-finite values")
        out[loc] = inv(to_flusight_levels(levels, q))
    return out


def run_components(pool: ModelPool, names, datasets, warnings: list[str]):
    """datasets: {key: (series, n_steps)} -> {key: {model: {location: array(n_steps, 23)}}}."""
    results = {key: {} for key in datasets}
    for name in names:
        try:
            model = pool.get(name)
        except Exception as e:
            log.exception("%s failed to load", name)
            warnings.append(f"{name} failed to load ({type(e).__name__}: {e}); left out")
            continue
        for key, (series, n_steps) in datasets.items():
            log.info("Running %s on %s data (%d steps)", name, key, n_steps)
            try:
                results[key][name] = _predict(model, series, n_steps)
            except Exception as e:
                log.exception("%s failed on %s data", name, key)
                warnings.append(f"{name} failed on {key} data ({type(e).__name__}: {e}); left out")
    return results


def ensemble(components) -> dict:
    names = list(components)
    if C.ENSEMBLE_METHOD == "weighted":
        w = np.array([float(C.ENSEMBLE_WEIGHTS.get(n, 0.0)) for n in names])
        if w.sum() <= 0:
            raise ValueError("ENSEMBLE_WEIGHTS gives no weight to the models that ran")
        agg = lambda a: np.tensordot(w / w.sum(), a, axes=1)
    elif C.ENSEMBLE_METHOD == "mean":
        agg = lambda a: a.mean(axis=0)
    else:
        agg = lambda a: np.median(a, axis=0)
    locs = components[names[0]].keys()
    return {loc: np.maximum(np.sort(agg(np.stack([components[n][loc] for n in names])), axis=-1), 0)
            for loc in locs}


def scale_spread(ens, last_obs, ref_date, offset):
    """Stretch each step's quantiles around the median on the log scale by SPREAD_SCALE[horizon]."""
    if all(float(v) == 1.0 for v in C.SPREAD_SCALE.values()):
        return ens
    out = {}
    for loc, q in ens.items():
        q = q.copy()
        for k in range(q.shape[0]):
            h = min(max(weeks_between(ref_date, last_obs + pd.Timedelta(weeks=k + 1)), 0), 3)
            s = float(C.SPREAD_SCALE.get(h, 1.0))
            lq = np.log(q[k] + offset)
            q[k] = np.maximum(np.exp(lq[C.MEDIAN_IDX] + s * (lq - lq[C.MEDIAN_IDX])) - offset, 0)
        out[loc] = q
    return out


# ------------------------------------------------------------- derived targets
def horizon_quantiles(ens, nowcasts: pd.DataFrame, last_obs, ref_date, offset):
    """{(location, horizon): array(23)}; already-reported weeks come from the nowcast."""
    z = _z(C.QUANTILES)
    now = nowcasts.set_index(["location", "date"])
    out = {}
    for loc, q in ens.items():
        for h in C.HORIZONS:
            ted = pd.Timestamp(ref_date) + pd.Timedelta(weeks=h)
            if ted <= last_obs:
                mu, sd = now.loc[(loc, ted), ["mu", "sd"]]
                out[loc, h] = np.maximum(np.exp(mu + z * sd) - offset, 0)
            else:
                out[loc, h] = q[weeks_between(last_obs, ted) - 1]
    return out


def rate_change_pmf(qv: np.ndarray, baseline: float, stable: float, large: float) -> np.ndarray:
    """Category probabilities for (target week - baseline week) from a quantile function."""
    stable = max(stable, C.MIN_STABLE_COUNT)
    large = max(large, stable)
    x = np.maximum.accumulate(qv) + np.arange(len(qv)) * 1e-9  # strictly increasing for interp
    edges = baseline + np.array([-large, -stable, stable, large])
    cdf = np.interp(edges, x, C.QUANTILES, left=0.0, right=1.0)
    p = np.clip(np.diff(np.concatenate([[0.0], cdf, [1.0]])), 0, None)
    return p / p.sum()


def sample_season(q, observed, offset_steps, n_future, rng) -> np.ndarray:
    """Sampled full-season trajectories, shape (N_PEAK_SAMPLES, n_observed + n_future).

    Each future week's marginal is the ensemble quantile function; weeks are tied
    together by a Gaussian copula with AR(1) correlation PEAK_RHO.
    """
    z_levels, n = _z(C.QUANTILES), C.N_PEAK_SAMPLES
    z = np.empty((n, n_future))
    z[:, 0] = rng.standard_normal(n)
    for t in range(1, n_future):
        z[:, t] = C.PEAK_RHO * z[:, t - 1] + np.sqrt(1 - C.PEAK_RHO**2) * rng.standard_normal(n)
    future = np.column_stack([np.interp(z[:, t], z_levels, q[offset_steps + t]) for t in range(n_future)])
    return np.hstack([np.broadcast_to(observed, (n, len(observed))), future])


def peak_distribution(ens, revised, last_obs, rng, season) -> dict:
    weeks = pd.date_range(season[0], season[1], freq=FREQ)
    past, future = weeks[weeks <= last_obs], weeks[weeks > last_obs]
    offset = weeks_between(last_obs, future[0]) - 1
    out = {}
    for loc, q in ens.items():
        observed = revised[loc].reindex(past).fillna(0).to_numpy()
        paths = sample_season(q, observed, offset, len(future), rng)
        out[loc] = {
            "weeks": weeks,
            "probs": np.bincount(paths.argmax(axis=1), minlength=len(weeks)) / C.N_PEAK_SAMPLES,
            "size": np.quantile(paths.max(axis=1), C.QUANTILES),
        }
    return out


# ---------------------------------------------------------------------- output
def _quantile_rows(target, hq, ref_date, cap, integer):
    rows = []
    for (loc, h), qv in hq.items():
        ted = (pd.Timestamp(ref_date) + pd.Timedelta(weeks=h)).date()
        v = np.minimum(qv, cap[loc])
        vals = [int(x) for x in np.rint(v)] if integer else [round(float(x), 6) for x in v]
        rows += [(ref_date, target, h, ted, loc, "quantile", f"{p:g}", x) for p, x in zip(C.QUANTILES, vals)]
    return rows


def _rate_change_rows(hq, locations, ref_date):
    locs = locations.set_index("location")
    rows = []
    for loc in sorted({loc for loc, _ in hq}):
        baseline = hq[loc, -1][C.MEDIAN_IDX]
        for h in C.RATE_CHANGE_HORIZONS:
            ted = (pd.Timestamp(ref_date) + pd.Timedelta(weeks=h)).date()
            s_col, l_col = C.RATE_CHANGE_COLS[h]
            pmf = rate_change_pmf(hq[loc, h], baseline, locs.at[loc, s_col], locs.at[loc, l_col])
            rows += [(ref_date, C.RATE_TARGET, h, ted, loc, "pmf", cat, float(p))
                     for cat, p in zip(C.RATE_CHANGE_CATEGORIES, pmf)]
    return rows


def _peak_rows(peaks, ref_date, cap):
    rows = []
    for loc, pk in peaks.items():
        rows += [(ref_date, C.PEAK_WEEK_TARGET, None, None, loc, "pmf", str(w.date()), float(p))
                 for w, p in zip(pk["weeks"], pk["probs"])]
        size = np.rint(np.minimum(pk["size"], cap[loc]))
        rows += [(ref_date, C.PEAK_INC_TARGET, None, None, loc, "quantile", f"{p:g}", int(v))
                 for p, v in zip(C.QUANTILES, size)]
    return rows


def make_frame(rows) -> pd.DataFrame:
    cols = ["reference_date", "target", "horizon", "target_end_date", "location",
            "output_type", "output_type_id", "value"]
    df = pd.DataFrame(rows, columns=cols)
    df["horizon"] = df["horizon"].astype("Int64")  # peak targets leave horizon blank
    return df.sort_values(["target", "location", "horizon"], kind="stable").reset_index(drop=True)


def validate(df: pd.DataFrame, hosp_locations) -> list[str]:
    """Basic checks; the hub's own validation runs on the pull request."""
    problems = []
    keys = ["target", "location", "horizon"]
    q = df[df["output_type"] == "quantile"]
    for key, g in q.groupby(keys, dropna=False):
        v = g.assign(level=g["output_type_id"].astype(float)).sort_values("level")["value"].to_numpy(float)
        if len(v) != len(C.QUANTILES):
            problems.append(f"{key}: {len(v)} quantiles, expected {len(C.QUANTILES)}")
        if (np.diff(v) < 0).any() or (v < 0).any():
            problems.append(f"{key}: quantiles not non-decreasing/non-negative")
    if (q.loc[q["target"] == C.ED_TARGET, "value"] > 1).any():
        problems.append("ED visit proportions above 1")
    hosp = q[q["target"] == C.HOSP_TARGET]
    for loc in hosp_locations:
        if set(hosp.loc[hosp["location"] == loc, "horizon"]) != set(C.HORIZONS):
            problems.append(f"{C.HOSP_TARGET}: missing horizons for {loc}")
    for key, g in df[df["output_type"] == "pmf"].groupby(keys, dropna=False):
        if abs(g["value"].sum() - 1) > 1e-6:
            problems.append(f"{key}: pmf does not sum to 1")
    return problems


# ------------------------------------------------------------------------ main
def run(ref_date: dt.date, frames: dict, locations: pd.DataFrame, models, targets,
        vintages: dict | None = None, prior_dir: Path | None = None, pool: ModelPool | None = None) -> Result:
    """Build the full submission for one reference date.

    frames: {"hosp": df, "ed": df} target data as it stood for this reference date.
    vintages: {key: {ref_date: df}} past data versions, for the nowcast's revision history.
    prior_dir: folder holding earlier submissions (model-output/<model_id>/), for the nowcast prior.
    """
    pool = pool or ModelPool()
    warnings: list[str] = []
    targets = set(targets) | {"hosp"}
    season = season_bounds(ref_date)
    if "peak" in targets and ref_date > season[1]:
        warnings.append("Reference date is after the season end; peak targets skipped")
        targets.discard("peak")
    keys = ["hosp"] + (["ed"] if "ed" in targets and "ed" in frames else [])

    reported, revised, nowcasts, last_obs, datasets = {}, {}, {}, {}, {}
    last_target_week = pd.Timestamp(ref_date) + pd.Timedelta(weeks=max(C.HORIZONS))
    for key in keys:
        series, lo, skipped = build_series(frames[key], locations, ref_date)
        if skipped:
            warnings.append(f"{key}: no recent data for {', '.join(skipped)}; omitted")
        if lo < pd.Timestamp(ref_date) - pd.Timedelta(weeks=1):
            warnings.append(f"{key}: data ends {lo.date()}; this week's update may not be posted yet")
        offset = C.LOG_OFFSET[key]
        rev_models = nc.fit_revisions((vintages or {}).get(key, {}), ref_date, offset,
                                      C.NOWCAST_MODEL.get(key, "constant")) if C.NOWCAST else {}
        dates = [lo - pd.Timedelta(weeks=k) for k in range(C.NOWCAST_WEEKS)]
        prior = nc.prior_forecasts(prior_dir, C.TARGET_NAMES[key], dates, offset) if C.NOWCAST else {}
        revised[key], nowcasts[key] = nc.nowcast(series, rev_models, prior, offset, adjust=C.NOWCAST)
        reported[key], last_obs[key] = series, lo
        n_steps = weeks_between(lo, last_target_week)
        if key == "hosp" and "peak" in targets:
            n_steps = max(n_steps, weeks_between(lo, season[1]))
        datasets[key] = (revised[key], n_steps)
        log.info("%s: data through %s, %d locations, %d steps", key, lo.date(), len(series), n_steps)

    components = run_components(pool, models, datasets, warnings)
    ensembles, horizon_q, peaks, rows = {}, {}, {}, []
    pop_cap = (locations.set_index("location")["population"] * C.MAX_POP_FRACTION).to_dict()
    problems = []
    for key in keys:
        if len(components[key]) < C.MIN_MODELS:
            msg = f"only {len(components[key])} model(s) succeeded on {key} data (need {C.MIN_MODELS})"
            (problems if key == "hosp" else warnings).append(msg + ("" if key == "hosp" else "; target skipped"))
            continue
        offset = C.LOG_OFFSET[key]
        ens = scale_spread(ensemble(components[key]), last_obs[key], ref_date, offset)
        ensembles[key] = ens
        hq = horizon_quantiles(ens, nowcasts[key], last_obs[key], ref_date, offset)
        horizon_q[key] = hq
        if key == "hosp":
            rows += _quantile_rows(C.HOSP_TARGET, hq, ref_date, pop_cap, integer=True)
            if "rate_change" in targets:
                rows += _rate_change_rows(hq, locations, ref_date)
            if "peak" in targets:
                peaks = peak_distribution(ens, revised[key], last_obs[key], np.random.default_rng(C.SEED), season)
                rows += _peak_rows(peaks, ref_date, pop_cap)
        else:
            if any((v > C.ED_MAX).any() for v in hq.values()):
                warnings.append(f"some ED quantiles exceeded {C.ED_MAX} and were capped")
            rows += _quantile_rows(C.ED_TARGET, hq, ref_date, {loc: C.ED_MAX for loc in ens}, integer=False)

    submission = make_frame(rows)
    if not problems:
        problems = validate(submission, reported["hosp"])
    return Result(ref_date, submission, problems, warnings, last_obs, reported, revised, nowcasts,
                  components, ensembles, horizon_q, peaks)


def archive(root: Path, ref) -> Path | None:
    """Move an earlier run's outputs for `ref` to root/archive/<ref>/<timestamp>/, keeping their layout."""
    root = Path(root)
    old = [root / "components" / str(ref), root / "model-output" / C.model_id() / f"{ref}-{C.model_id()}.csv",
           *(root / "nowcasts").glob(f"{ref}-*"), *(root / "reports").glob(f"{ref}-*")]
    old = [p for p in old if p.exists()]
    if not old:
        return None
    dest = root / "archive" / str(ref) / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    for p in old:
        target = dest / p.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        p.rename(target)
    log.info("Moved the earlier run for %s to %s", ref, dest)
    return dest


def save(result: Result, root: Path) -> Path | None:
    """Write the submission (if valid), component forecasts and nowcasts under `root`, archiving any earlier run."""
    root = Path(root)
    ref = result.ref_date
    archive(root, ref)
    for key, comps in result.components.items():
        rows = []
        for name, preds in comps.items():
            for loc, q in preds.items():
                for k in range(q.shape[0]):
                    ted = result.last_obs[key] + pd.Timedelta(weeks=k + 1)
                    if ted > pd.Timestamp(ref) + pd.Timedelta(weeks=max(C.HORIZONS)):
                        break
                    rows += [(name, loc, ted, f"{p:g}", float(v)) for p, v in zip(C.QUANTILES, q[k])]
        d = root / "components" / str(ref)
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["model", "location", "target_end_date", "quantile", "value"]).to_parquet(
            d / f"{key}.parquet", index=False)
    for key, table in result.nowcasts.items():
        d = root / "nowcasts"
        d.mkdir(parents=True, exist_ok=True)
        table.to_csv(d / f"{ref}-{key}.csv", index=False)
    if result.problems:
        log.error("Not writing a submission file: %s", "; ".join(result.problems))
        return None
    path = root / "model-output" / C.model_id() / f"{ref}-{C.model_id()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    result.submission.to_csv(path, index=False)
    log.info("Wrote %s (%d rows)", path, len(result.submission))
    return path
