"""Scoring forecasts against the latest data.

- WIS (weighted interval score) on the natural scale and on log(x + offset).
  For FluSight's 23 quantiles, WIS = 2 x mean pinball loss.
- 50% and 95% interval coverage.
- Relative WIS against the hub's FluSight-baseline on the same forecasts.
- Brier score for the rate-change target.
- Scores for each component model, and for every combination of components,
  from the component forecasts saved alongside each submission.
"""
from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as C
from . import data

log = logging.getLogger(__name__)
KEYS = ["model", "reference_date", "target", "horizon", "location"]


def _offsets() -> dict:
    return {C.TARGET_NAMES[k]: v for k, v in C.LOG_OFFSET.items()}


def truth_table() -> pd.DataFrame:
    frames = []
    for key, target in C.TARGET_NAMES.items():
        df = data.load_current(key).dropna(subset=["value"]).drop_duplicates(["location", "date"], keep="last")
        frames.append(df.assign(target=target).rename(columns={"date": "target_end_date", "value": "truth"}))
    return pd.concat(frames, ignore_index=True)


def load_submissions(root: Path, model: str | None = None) -> pd.DataFrame:
    model = model or C.model_id()
    files = sorted((Path(root) / "model-output" / model).glob(f"*-{model}.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, dtype={"location": str, "output_type_id": str}) for f in files])
    return _parse(df.assign(model=model))


def _parse(df: pd.DataFrame) -> pd.DataFrame:
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df["target_end_date"] = pd.to_datetime(df["target_end_date"])
    df["horizon"] = df["horizon"].astype("Int64")
    return df


def load_baseline(ref_dates) -> pd.DataFrame:
    """The hub's FluSight-baseline submissions for these reference dates (cached)."""
    d = C.CACHE_DIR / "baseline"
    d.mkdir(parents=True, exist_ok=True)
    frames = []
    for ref in sorted({pd.Timestamp(r).date() for r in ref_dates}):
        name = f"{ref}-{C.BASELINE_MODEL}.csv"
        f, missing = d / name, d / (name + ".missing")
        if not f.exists() and not missing.exists():
            try:
                f.write_text(data._get(C.hub_raw(f"model-output/{C.BASELINE_MODEL}/{name}")).text, encoding="utf-8")
            except requests.HTTPError:
                if (pd.Timestamp.today() - pd.Timestamp(ref)).days > 14:  # recent ones may just not be posted yet
                    missing.touch()
            except requests.RequestException as e:
                log.warning("Could not download %s: %s", name, e)
        if f.exists():
            frames.append(pd.read_csv(f, dtype={"location": str, "output_type_id": str}))
    if not frames:
        return pd.DataFrame()
    return _parse(pd.concat(frames).assign(model=C.BASELINE_MODEL))


def _pinball(q, y, tau):
    return (tau - (y < q)) * (y - q)


def score_quantiles(fc: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, reference_date, target, horizon, location) with WIS and coverage."""
    q = fc[(fc["output_type"] == "quantile") & fc["horizon"].notna()]
    df = q.merge(truth, on=["target", "location", "target_end_date"])
    if df.empty:
        return pd.DataFrame(columns=KEYS + ["wis", "wis_log", "cov50", "cov95", "truth", "median"])
    tau = df["output_type_id"].astype(float)
    val, y = df["value"].astype(float), df["truth"].astype(float)
    off = df["target"].map(_offsets())
    df = df.assign(tau=tau, pin=_pinball(val, y, tau),
                   pin_log=_pinball(np.log(val.clip(lower=0) + off), np.log(y + off), tau))
    g = df.groupby(KEYS)
    out = (g[["pin", "pin_log"]].mean() * 2).rename(columns={"pin": "wis", "pin_log": "wis_log"})
    wide = df.pivot_table(index=KEYS, columns="tau", values="value")
    level = lambda p: wide[min(wide.columns, key=lambda c: abs(c - p))]
    t = g["truth"].first()
    out["truth"] = t
    out["median"] = level(0.5)
    out["cov50"] = ((level(0.25) <= t) & (t <= level(0.75))).astype(float)
    out["cov95"] = ((level(0.025) <= t) & (t <= level(0.975))).astype(float)
    return out.reset_index()


def score_rate_change(fc: pd.DataFrame, truth: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    """Brier score (sum over the 5 categories) for the rate-change target."""
    p = fc[fc["target"] == C.RATE_TARGET]
    if p.empty:
        return pd.DataFrame()
    hosp = truth[truth["target"] == C.HOSP_TARGET].set_index(["location", "target_end_date"])["truth"]
    locs = locations.set_index("location")
    rows = []
    for (model, ref, h, loc), g in p.groupby(["model", "reference_date", "horizon", "location"]):
        ted, base_date = ref + pd.Timedelta(weeks=int(h)), ref - pd.Timedelta(weeks=1)
        if (loc, ted) not in hosp.index or (loc, base_date) not in hosp.index:
            continue
        diff = hosp[loc, ted] - hosp[loc, base_date]
        s_col, l_col = C.RATE_CHANGE_COLS[int(h)]
        stable = max(locs.at[loc, s_col], C.MIN_STABLE_COUNT)
        large = max(locs.at[loc, l_col], stable)
        cat = ("stable" if abs(diff) < stable else
               ("large_" if abs(diff) >= large else "") + ("increase" if diff > 0 else "decrease"))
        probs = g.set_index("output_type_id")["value"].reindex(C.RATE_CHANGE_CATEGORIES).fillna(0)
        outcome = (probs.index == cat).astype(float)
        rows.append((model, ref, C.RATE_TARGET, h, loc, float(((probs - outcome) ** 2).sum()), cat))
    return pd.DataFrame(rows, columns=KEYS + ["brier", "observed"])


def relative_summary(scores: pd.DataFrame, by=("target", "horizon")) -> pd.DataFrame:
    """Mean scores per model, with WIS relative to the baseline on the forecasts both made."""
    by = list(by)
    base = scores[scores["model"] == C.BASELINE_MODEL].drop(columns="model")
    rows = []
    for model, g in scores.groupby("model"):
        j = g.merge(base[KEYS[1:] + ["wis", "wis_log"]], on=KEYS[1:], how="left", suffixes=("", "_base"))
        for key, gg in j.groupby(by):
            both = gg.dropna(subset=["wis_base"])
            rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,))) | {
                "model": model,
                "n": len(gg),
                "wis": gg["wis"].mean(),
                "wis_log": gg["wis_log"].mean(),
                "rel_wis": both["wis"].sum() / both["wis_base"].sum() if len(both) else np.nan,
                "rel_wis_log": both["wis_log"].sum() / both["wis_log_base"].sum() if len(both) else np.nan,
                "cov50": gg["cov50"].mean(),
                "cov95": gg["cov95"].mean(),
            })
    return pd.DataFrame(rows)


def load_components(root: Path) -> pd.DataFrame:
    frames = []
    for f in sorted((Path(root) / "components").glob("*/*.parquet")):
        ref, key = pd.Timestamp(f.parent.name), f.stem
        df = pd.read_parquet(f)
        df["target_end_date"] = pd.to_datetime(df["target_end_date"])
        frames.append(df.assign(reference_date=ref, target=C.TARGET_NAMES[key]))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["horizon"] = ((df["target_end_date"] - df["reference_date"]).dt.days // 7).astype("Int64")
    return df[df["horizon"].between(0, max(C.HORIZONS))].rename(columns={"quantile": "output_type_id"}).assign(
        output_type="quantile")


def combination_scores(components: pd.DataFrame, truth: pd.DataFrame, baseline_scores: pd.DataFrame) -> pd.DataFrame:
    """Score every subset (2+ models) of components under median and mean ensembling."""
    if components.empty:
        return pd.DataFrame()
    idx = ["reference_date", "target", "horizon", "location", "target_end_date", "output_type_id"]
    wide = components.pivot_table(index=idx, columns="model", values="value")
    names = list(wide.columns)
    rows = []
    for r in range(2, len(names) + 1):
        for subset in itertools.combinations(names, r):
            sub = wide[list(subset)].dropna()
            for method in ("median", "mean"):
                vals = sub.median(axis=1) if method == "median" else sub.mean(axis=1)
                fc = vals.rename("value").reset_index().assign(output_type="quantile", model="ens")
                s = score_quantiles(fc, truth)
                label = f"{method}: {' + '.join(subset)}"
                summary = relative_summary(pd.concat([s, baseline_scores]), by=("target",))
                for _, row in summary[summary["model"] == "ens"].iterrows():
                    rows.append(row.to_dict() | {"model": label, "method": method, "n_models": r})
    return pd.DataFrame(rows).sort_values(["target", "rel_wis_log"])


def evaluate(root: Path, combinations: bool = False) -> dict[str, pd.DataFrame]:
    """Score everything under `root` (forecasts/ or a backtest folder); writes CSVs to root/evaluation/."""
    root = Path(root)
    subs = load_submissions(root)
    if subs.empty:
        log.info("No submissions under %s to evaluate", root)
        return {}
    truth = truth_table()
    base = load_baseline(subs["reference_date"].unique())
    comps = load_components(root)
    all_fc = pd.concat([subs, base] + ([comps] if not comps.empty else []), ignore_index=True)
    scores = score_quantiles(all_fc, truth)
    out = {
        "scores": scores,
        "by_horizon": relative_summary(scores),
        "by_week": relative_summary(scores, by=("target", "reference_date")),
        "by_location": relative_summary(scores, by=("target", "location")),
    }
    locations = data.load_locations()
    rc = score_rate_change(pd.concat([subs, base], ignore_index=True), truth, locations)
    if not rc.empty:
        out["rate_change"] = rc.groupby(["model", "horizon"])["brier"].agg(["mean", "count"]).reset_index()
    if combinations:
        out["combinations"] = combination_scores(comps, truth, scores[scores["model"] == C.BASELINE_MODEL])
    d = root / "evaluation"
    d.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(d / f"{name}.csv", index=False)
    log.info("Wrote evaluation tables to %s", d)
    return out
