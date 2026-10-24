"""Season test: replay a past season week by week, then tabulate how each week went.

Each week is forecast with only what was available at that week's deadline:
- the version of the hub's target data committed before that Wednesday,
- revision history learned only from data versions up to then, and
- the test's own previous forecasts as the nowcast prior.

Forecasts are then scored against the data as it stands today. Results go to
backtests/<name>/evaluation/weekly_results.{csv,md,html}.

Caveat: the foundation models were pretrained by their developers, and their
training data may include some of the test season. Zero-shot results can
therefore look better than they would have in real time. That applies most to
models released after the season (e.g. Timestral, Oct 2026).
"""
from __future__ import annotations

import datetime as dt
import html
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import backtest, evaluate
from .forecast import season_bounds

log = logging.getLogger(__name__)


def last_season(today: dt.date | None = None) -> tuple[dt.date, dt.date]:
    """Start and end Saturdays of the most recently completed season."""
    today = today or dt.date.today()
    year = today.year - 1 if today.month >= 8 else today.year - 2
    return season_bounds(dt.date(year, 10, 15))


def _rel(num: pd.Series, den: pd.Series) -> float:
    return float(num.sum() / den.sum()) if len(den) and den.sum() > 0 else np.nan


def _summarize(g: pd.DataFrame, comps: pd.DataFrame) -> dict:
    """Scores for one week (or the whole season) of our forecasts, horizons 0-3."""
    fc = g[g["horizon"] >= 0]
    both = fc.dropna(subset=["wis_base"])
    row = {
        "locations": fc["location"].nunique(),
        "wis": fc["wis"].mean(),
        "baseline_wis": both["wis_base"].mean() if len(both) else np.nan,
        "rel_wis": _rel(both["wis"], both["wis_base"]),
        "rel_wis_log": _rel(both["wis_log"], both["wis_log_base"]),
        "cov50": fc["cov50"].mean(),
        "cov95": fc["cov95"].mean(),
        "h-1_wis": g.loc[g["horizon"] == -1, "wis"].mean(),
    }
    for model, cg in comps.groupby("model"):
        cb = cg.dropna(subset=["wis_log_base"])
        row[f"rel_wis_log {model}"] = _rel(cb["wis_log"], cb["wis_log_base"])
    return row


def weekly_table(root: Path, ev: dict) -> pd.DataFrame:
    me, base_name = C.model_id(), C.BASELINE_MODEL
    keys = evaluate.KEYS[1:]
    s = ev["scores"]
    s = s[s["target"] == C.HOSP_TARGET]
    base = s[s["model"] == base_name][keys + ["wis", "wis_log"]]
    with_base = s.merge(base, on=keys, how="left", suffixes=("", "_base"))
    ours = with_base[with_base["model"] == me]
    comps = with_base[~with_base["model"].isin([me, base_name])]
    truth = evaluate.truth_table()
    truth = truth[truth["target"] == C.HOSP_TARGET].set_index(["location", "target_end_date"])["truth"]

    rows = []
    for ref, g in ours.groupby("reference_date"):
        row = {"reference_date": str(ref.date())}
        nf = root / "nowcasts" / f"{ref.date()}-hosp.csv"
        if nf.exists():
            now = pd.read_csv(nf, dtype={"location": str}, parse_dates=["date"])
            now = now[now["lag"] == 0]
            row["data_through"] = str(now["date"].max().date())
            t = now.set_index(["location", "date"]).join(truth.rename_axis(["location", "date"]), how="inner")
            if len(t):
                err = lambda col: float(np.mean(np.abs(np.log(t[col] + 1) - np.log(t["truth"] + 1))))
                row["report_err"], row["nowcast_err"] = err("reported"), err("median")
        cf = root / "components" / str(ref.date()) / "hosp.parquet"
        if cf.exists():
            row["models"] = ", ".join(sorted(pd.read_parquet(cf, columns=["model"])["model"].unique()))
        us = g[(g["location"] == "US") & (g["horizon"] == 1)]
        if len(us):
            row["US h1 median"], row["US h1 observed"] = float(us["median"].iloc[0]), float(us["truth"].iloc[0])
        rows.append(row | _summarize(g, comps[comps["reference_date"] == ref]))
    table = pd.DataFrame(rows)
    if len(table):
        season = {"reference_date": "SEASON"} | _summarize(ours, comps)
        if {"report_err", "nowcast_err"} <= set(table):
            season["report_err"], season["nowcast_err"] = table["report_err"].mean(), table["nowcast_err"].mean()
        table = pd.concat([table, pd.DataFrame([season])], ignore_index=True)
    front = ["reference_date", "data_through", "models", "locations", "wis", "baseline_wis", "rel_wis",
             "rel_wis_log", "cov50", "cov95", "h-1_wis", "report_err", "nowcast_err", "US h1 median", "US h1 observed"]
    return table[[c for c in front if c in table] + [c for c in table if c not in front]]


COLUMN_NOTES = {
    "data_through": "last reported week in the data available that week",
    "wis": "mean WIS of our forecasts, horizons 0-3, all locations",
    "baseline_wis": "same for FluSight-baseline (forecasts both made)",
    "rel_wis": "our WIS / baseline WIS (below 1 = better than baseline)",
    "rel_wis_log": "same on log(x+1) scale, so large locations don't dominate",
    "cov50 / cov95": "share of observations inside our 50% / 95% intervals",
    "h-1_wis": "WIS of the horizon -1 (nowcast) forecast",
    "report_err / nowcast_err": "mean |log error| vs final data of the latest week as reported vs as nowcast",
    "rel_wis_log <model>": "each component model alone, relative to baseline",
}


def _fmt(col: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if not isinstance(v, (int, float, np.number)):
        return str(v)
    if col.startswith("cov"):
        return f"{v * 100:.0f}%"
    if col.startswith("rel_") or col.endswith("_err"):
        return f"{v:.2f}"
    if col == "locations":
        return f"{int(v)}"
    return f"{v:,.1f}" if col.endswith("wis") else f"{v:,.0f}"


def write_outputs(table: pd.DataFrame, root: Path, start, end, models) -> Path:
    d = root / "evaluation"
    d.mkdir(parents=True, exist_ok=True)
    table.to_csv(d / "weekly_results.csv", index=False)
    cells = [[_fmt(c, v) for c, v in zip(table.columns, r)] for r in table.itertuples(index=False)]

    md = [f"# Season test {start} to {end}", "",
          f"Model `{C.model_id()}`, components: {', '.join(models)}. Each week uses only data available at that week's deadline; "
          "scored against current data.", "",
          "| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    md += ["| " + " | ".join(r) + " |" for r in cells]
    md += ["", "Columns:"] + [f"- **{k}**: {v}" for k, v in COLUMN_NOTES.items()]
    (d / "weekly_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    head = "".join(f"<th>{html.escape(c)}</th>" for c in table.columns)
    body = "".join("<tr" + (' class="total"' if r[0] == "SEASON" else "") + ">" +
                   "".join(f"<td>{html.escape(x)}</td>" for x in r) + "</tr>" for r in cells)
    notes = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>" for k, v in COLUMN_NOTES.items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Season Test Results</title>
<style>
:root {{ color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e1e0d9; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1400px; margin:0 auto; padding:24px 16px; }}
.wrap {{ overflow-x:auto; background:var(--surface); border:1px solid var(--grid); border-radius:8px; }}
table {{ border-collapse:collapse; font-size:13px; width:100%; }}
th, td {{ padding:6px 8px; border-bottom:1px solid var(--grid); text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }}
th {{ color:var(--ink2); position:sticky; top:0; background:var(--surface); }}
th:first-child, td:first-child {{ text-align:left; }}
tr.total td {{ font-weight:700; border-top:2px solid var(--ink2); }}
p, li {{ color:var(--ink2); }}
</style></head><body><main>
<h1>Season test: {start} to {end}</h1>
<p>Model {html.escape(C.model_id())}; components: {html.escape(', '.join(models))}. Each week was forecast using only the data
available at that week's deadline, then scored against the data as it stands now.</p>
<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
<ul>{notes}</ul>
<p>Model pretraining data may overlap the test season, which would flatter zero-shot results.</p>
</main></body></html>"""
    out = d / "weekly_results.html"
    out.write_text(page, encoding="utf-8")
    return out


def run(start: dt.date, end: dt.date, name: str, models, targets, every: int = 1) -> Path:
    ev = backtest.run(start, end, name, models, targets, every=every, resume=True)
    if not ev:
        raise RuntimeError("the season test produced no forecasts to score")
    root = C.BACKTEST_DIR / name
    table = weekly_table(root, ev)
    out = write_outputs(table, root, start, end, models)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_rows", 100):
        print(table.round(3).to_string(index=False))
    return out
