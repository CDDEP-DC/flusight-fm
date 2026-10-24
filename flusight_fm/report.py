"""Pre-submission review report: a self-contained HTML page per reference date.

The page is written to <root>/reports/<ref>-report.html. It works offline
(no external scripts) and can be opened on any machine or emailed. The data
behind it is saved next to it as <ref>-data.json, so `flusight report` can
re-render it with up-to-date evaluation scores without rerunning the models.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import evaluate
from .forecast import Result, weeks_between

log = logging.getLogger(__name__)
TEMPLATE = Path(__file__).with_name("report_template.html")
BANDS = {"q025": 0.025, "q25": 0.25, "q50": 0.5, "q75": 0.75, "q975": 0.975}
LABELS = {"hosp": "Hospital admissions", "ed": "ED visits (% flu)"}


def _iso(ts) -> str:
    return str(pd.Timestamp(ts).date())


def _num(key, v):
    if v is None or not np.isfinite(v):
        return None
    return round(float(v), 1) if key == "hosp" else round(float(v), 6)


def _idx(level: float) -> int:
    return int(np.argmin(np.abs(C.QUANTILES - level)))


def _previous_submission(root: Path, ref: dt.date) -> pd.DataFrame:
    f = root / "model-output" / C.model_id() / f"{ref - dt.timedelta(weeks=1)}-{C.model_id()}.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f, dtype={"location": str, "output_type_id": str})
    return df[(df["output_type"] == "quantile") & (df["output_type_id"] == "0.5")]


def _checks(result: Result, prev: pd.DataFrame) -> list[dict]:
    checks = [{"level": "error", "text": p} for p in result.problems]
    checks += [{"level": "warning", "text": w} for w in result.warnings]
    hq = result.horizon_q.get("hosp", {})
    now = result.nowcasts.get("hosp")
    if now is not None and not now.empty:
        big = now[(now["lag"] == 0) & (now["reported"] >= 20) & ((now["median"] / now["reported"] - 1).abs() > 0.15)]
        for _, r in big.iterrows():
            checks.append({"level": "info", "text": f"Nowcast revises {r['location']} {_iso(r['date'])}: "
                                                    f"reported {r['reported']:.0f} → {r['median']:.0f}"})
    if not prev.empty and hq:
        p = prev[(prev["target"] == C.HOSP_TARGET) & (prev["horizon"] == 1)].set_index("location")["value"]
        for loc, old in p.items():
            if (loc, 0) in hq:
                new = hq[loc, 0][C.MEDIAN_IDX]
                if max(new, old) >= 20 and abs(new / max(old, 1) - 1) > 0.5:
                    checks.append({"level": "warning", "text": f"{loc}: this week's horizon-0 median ({new:.0f}) differs "
                                                               f"from last week's horizon-1 median ({old:.0f}) by >50%"})
    comps = result.components.get("hosp", {})
    if len(comps) >= 2:
        k = weeks_between(result.last_obs["hosp"], pd.Timestamp(result.ref_date) + pd.Timedelta(weeks=1)) - 1
        for loc in result.reported["hosp"]:
            meds = [c[loc][k][C.MEDIAN_IDX] for c in comps.values() if loc in c]
            if np.median(meds) >= 20 and max(meds) / max(min(meds), 1) > 3:
                checks.append({"level": "warning", "text": f"{loc}: component models disagree at horizon 1 "
                                                           f"(medians {min(meds):.0f}–{max(meds):.0f})"})
    if not checks:
        checks.append({"level": "good", "text": "No problems found"})
    return checks


def build_data(result: Result, root: Path, locations: pd.DataFrame) -> dict:
    ref = result.ref_date
    prev = _previous_submission(root, ref)
    locs = locations.sort_values("population", ascending=False)
    targets = {}
    for key, hq in result.horizon_q.items():
        lo = result.last_obs[key]
        fdates = [_iso(pd.Timestamp(ref) + pd.Timedelta(weeks=h)) for h in C.HORIZONS]
        now = result.nowcasts[key]
        series, fc, comps, prior = {}, {}, {}, {}
        for loc, s in result.reported[key].items():
            s = s.iloc[-104:]
            nl = now[now["location"] == loc].sort_values("date")
            series[loc] = {"dates": [_iso(d) for d in s.index], "reported": [_num(key, v) for v in s],
                           "nowcast": {"dates": [_iso(d) for d in nl["date"]],
                                       "median": [_num(key, v) for v in nl["median"]]}}
            if (loc, 0) not in hq:
                continue
            fc[loc] = {"dates": fdates} | {b: [_num(key, hq[loc, h][_idx(lv)]) for h in C.HORIZONS]
                                           for b, lv in BANDS.items()}
            comps[loc] = {}
            for name, preds in result.components.get(key, {}).items():
                vals = []
                for h in C.HORIZONS:
                    k = weeks_between(lo, pd.Timestamp(ref) + pd.Timedelta(weeks=h)) - 1
                    vals.append(_num(key, preds[loc][k][C.MEDIAN_IDX]) if 0 <= k < len(preds[loc]) else None)
                comps[loc][name] = vals
        if not prev.empty:
            p = prev[prev["target"] == C.TARGET_NAMES[key]]
            for loc, g in p.groupby("location"):
                g = g.sort_values("target_end_date")
                prior[loc] = {"dates": list(g["target_end_date"]), "q50": [_num(key, v) for v in g["value"]]}
        targets[key] = {"label": LABELS[key], "series": series, "forecast": fc, "comps": comps, "prev": prior,
                        "models": list(result.components.get(key, {}))}

    rate = {}
    peak = {}
    sub = result.submission
    rc = sub[sub["target"] == C.RATE_TARGET]
    for (loc, h), g in rc.groupby(["location", "horizon"]):
        rate.setdefault(loc, {})[int(h)] = [round(float(v), 4) for v in
                                            g.set_index("output_type_id")["value"].reindex(C.RATE_CHANGE_CATEGORIES)]
    for loc, pk in result.peaks.items():
        peak[loc] = {"weeks": [_iso(w) for w in pk["weeks"]], "probs": [round(float(p), 4) for p in pk["probs"]],
                     "size": [round(float(pk["size"][_idx(lv)])) for lv in (0.025, 0.5, 0.975)]}

    return {
        "meta": {
            "model_id": C.model_id(),
            "ref_date": str(ref),
            "generated": dt.datetime.now().isoformat(timespec="minutes"),
            "machine": f"{platform.node()} ({platform.system()} {platform.machine()})",
            "data_through": {k: _iso(v) for k, v in result.last_obs.items()},
            "settings": {"ensemble": C.ENSEMBLE_METHOD, "transform": C.TRANSFORM, "nowcast": C.NOWCAST,
                         "nowcast_weeks": C.NOWCAST_WEEKS, "prior_weight": C.PRIOR_WEIGHT},
            "submission_written": not result.problems,
        },
        "checks": _checks(result, prev),
        "locations": [{"code": r.location, "name": r.location_name} for r in locs.itertuples()
                      if r.location in result.reported["hosp"]],
        "categories": C.RATE_CHANGE_CATEGORIES,
        "targets": targets,
        "rate": rate,
        "peak": peak,
    }


def _evaluation(root: Path) -> dict | None:
    try:
        ev = evaluate.evaluate(root)
    except Exception as e:  # evaluation is a convenience; never block the report on it
        log.warning("Evaluation skipped: %s", e)
        return None
    if not ev:
        return None
    bh = ev["by_horizon"]
    bw = ev["by_week"]
    bw = bw[bw["model"].isin([C.model_id(), C.BASELINE_MODEL])]
    clean = lambda df: json.loads(df.to_json(orient="records", date_format="iso"))
    return {"by_horizon": clean(bh), "by_week": clean(bw.sort_values("reference_date"))}


def render(payload: dict, out: Path) -> Path:
    blob = json.dumps(payload, default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", blob), encoding="utf-8")
    return out


def write(result: Result, root: Path, locations: pd.DataFrame) -> Path:
    root = Path(root)
    payload = build_data(result, root, locations)
    d = root / "reports"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{result.ref_date}-data.json").write_text(
        json.dumps(payload, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    payload["eval"] = _evaluation(root)
    path = render(payload, d / f"{result.ref_date}-report.html")
    log.info("Review report: %s", path)
    return path


def rerender(ref: dt.date, root: Path) -> Path:
    root = Path(root)
    d = root / "reports"
    payload = json.loads((d / f"{ref}-data.json").read_text(encoding="utf-8"))
    payload["eval"] = _evaluation(root)
    return render(payload, d / f"{ref}-report.html")
