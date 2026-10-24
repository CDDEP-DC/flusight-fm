"""Command line: `flusight <command>` (or `python -m flusight_fm <command>`).

Weekly:
    flusight forecast        build this week's forecast, nowcast and review report
    flusight submit          open a pull request to the hub (after reviewing the report)

Occasionally:
    flusight report          re-render a report with fresh evaluation scores
    flusight evaluate        score past forecasts (or a backtest) against current data
    flusight season-test     replay last season week by week and tabulate the weekly results
    flusight backtest        rerun the pipeline on past weeks, using the data as it was then
    flusight fetch-history   download NHSN (2020-) and ILINet (1997-) histories
    flusight finetune        fine-tune Chronos-2 on those histories
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

from . import config as C

log = logging.getLogger("flusight")


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def cmd_forecast(args) -> int:
    from . import data, forecast, report

    targets = args.targets or C.ALL_TARGETS
    models = args.models or C.MODELS
    current_ref = data.reference_date_for(dt.date.today())
    ref = args.reference_date or current_ref
    if ref.weekday() != 5:
        log.error("Reference date %s is not a Saturday", ref)
        return 2
    keys = ["hosp"] + (["ed"] if "ed" in targets else [])
    vintages = {}
    if C.NOWCAST or ref < current_ref:
        for key in keys:
            try:
                vintages[key] = data.load_vintages(key)
            except Exception as e:
                log.warning("Could not load past versions of %s data (%s); nowcast falls back to defaults", key, e)
    frames = {}
    for key in keys:
        if ref < current_ref and vintages.get(key):
            frames[key] = data.vintage_for(vintages[key], ref)
            log.info("Past reference date: using the %s data as it stood then", key)
        else:
            frames[key] = data.load_current(key)
    frames = {k: v for k, v in frames.items() if v is not None}
    locations = data.load_locations()
    result = forecast.run(ref, frames, locations, models, targets, vintages, prior_dir=args.out)
    for w in result.warnings:
        log.warning(w)
    path = forecast.save(result, args.out)
    if not args.no_report:
        html = report.write(result, args.out, locations)
        print(f"\nReview the forecast: {html.resolve().as_uri()}")
    if path:
        print(f"Submission file: {path}\nWhen it looks right: flusight submit --reference-date {ref}")
    return 0 if path else 1


def cmd_report(args) -> int:
    from . import data, report

    ref = args.reference_date or data.reference_date_for(dt.date.today())
    html = report.rerender(ref, args.out)
    print(html.resolve().as_uri())
    return 0


def cmd_evaluate(args) -> int:
    import pandas as pd

    from . import evaluate

    out = evaluate.evaluate(args.dir, combinations=args.combinations)
    if not out:
        print("Nothing to evaluate yet.")
        return 0
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.max_rows", 200):
        print(out["by_horizon"].round(3).to_string(index=False))
        if "combinations" in out:
            print("\nBest ensemble combinations (by relative log-WIS):")
            print(out["combinations"].head(15).round(3).to_string(index=False))
    return 0


def cmd_backtest(args) -> int:
    from . import backtest

    backtest.run(args.start, args.end, args.name, args.models or C.MODELS,
                 set(args.targets or ["hosp", "rate_change"]), every=args.every)
    return 0


def cmd_season_test(args) -> int:
    from . import season_test

    start, end = season_test.last_season()
    start, end = args.start or start, args.end or end
    name = args.name or f"season-{start.year}-{str(start.year + 1)[2:]}"
    out = season_test.run(start, end, name, args.models or C.MODELS,
                          set(args.targets or ["hosp", "rate_change"]), every=args.every)
    print(f"\nWeekly results: {out.resolve().as_uri()}")
    return 0


def cmd_fetch_history(args) -> int:
    from . import data

    locations = data.load_locations()
    nhsn = data.fetch_nhsn_history(locations, refresh=True)
    ili = data.fetch_ilinet_history(locations, refresh=True)
    print(f"NHSN: {len(nhsn)} rows from {nhsn['date'].min().date()}; ILINet: {len(ili)} rows from {ili['date'].min().date()}")
    return 0


def cmd_finetune(args) -> int:
    from . import finetune

    out = args.out or Path(C.CHRONOS2_FINETUNED).parent
    finetune.finetune_chronos2(args.cutoff, out, steps=args.steps, learning_rate=args.lr, mode=args.mode,
                               batch_size=args.batch_size, prediction_length=args.prediction_length,
                               include_ilinet=not args.no_ilinet)
    print(f"Saved to {out}. Use it as model 'chronos2_ft' (set chronos2_finetuned in flusight.toml if you "
          f"saved it somewhere else).")
    return 0


def cmd_submit(args) -> int:
    from . import data, submit

    ref = args.reference_date or data.reference_date_for(dt.date.today())
    csv = args.out / "model-output" / C.model_id() / f"{ref}-{C.model_id()}.csv"
    if not csv.exists():
        log.error("No submission file %s; run `flusight forecast` first", csv)
        return 1
    if not args.hub_dir:
        log.error("Set --hub-dir or FLUSIGHT_HUB_DIR to your local clone of your fork of the hub")
        return 2
    if not args.yes:
        if not sys.stdin.isatty():
            log.error("Not a terminal; pass --yes to submit without confirmation")
            return 2
        report = args.out / "reports" / f"{ref}-report.html"
        print(f"About to submit {csv.name} to {C.HUB_REPO}.\nReview report: {report.resolve().as_uri()}")
        if input("Submit? [y/N] ").strip().lower() != "y":
            print("Not submitted.")
            return 1
    submit.submit(csv, args.hub_dir, ref, args.include_metadata)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flusight", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=C.REPO / "flusight.toml", help="settings file (default: flusight.toml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("forecast", help="build a week's forecast and review report")
    f.add_argument("--reference-date", type=_date, help="Saturday reference date (default: this week's)")
    f.add_argument("--models", nargs="+", help="component models (default: settings)")
    f.add_argument("--targets", nargs="+", choices=C.ALL_TARGETS, help="targets (default: all)")
    f.add_argument("--out", type=Path, default=C.FORECAST_DIR, help="output folder (default: forecasts/)")
    f.add_argument("--no-report", action="store_true")
    f.set_defaults(func=cmd_forecast)

    r = sub.add_parser("report", help="re-render a review report with fresh evaluation scores")
    r.add_argument("--reference-date", type=_date)
    r.add_argument("--out", type=Path, default=C.FORECAST_DIR)
    r.set_defaults(func=cmd_report)

    e = sub.add_parser("evaluate", help="score past forecasts against current data")
    e.add_argument("--dir", type=Path, default=C.FORECAST_DIR, help="forecasts/ or backtests/<name>")
    e.add_argument("--combinations", action="store_true", help="also score every combination of component models")
    e.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("backtest", help="rerun past weeks with the data as it was then")
    b.add_argument("--start", type=_date, required=True)
    b.add_argument("--end", type=_date, required=True)
    b.add_argument("--name", required=True, help="output folder under backtests/")
    b.add_argument("--models", nargs="+")
    b.add_argument("--targets", nargs="+", choices=C.ALL_TARGETS, help="default: hosp rate_change")
    b.add_argument("--every", type=int, default=1, help="run every Nth week")
    b.set_defaults(func=cmd_backtest)

    st = sub.add_parser("season-test", help="replay last season week by week and tabulate weekly results")
    st.add_argument("--start", type=_date, help="first reference date (default: start of last season)")
    st.add_argument("--end", type=_date, help="last reference date (default: end of last season)")
    st.add_argument("--name", help="output folder under backtests/ (default: season-YYYY-YY)")
    st.add_argument("--models", nargs="+")
    st.add_argument("--targets", nargs="+", choices=C.ALL_TARGETS, help="default: hosp rate_change")
    st.add_argument("--every", type=int, default=1, help="run every Nth week (2 = quick pass)")
    st.set_defaults(func=cmd_season_test)

    h = sub.add_parser("fetch-history", help="download long NHSN and ILINet histories")
    h.set_defaults(func=cmd_fetch_history)

    t = sub.add_parser("finetune", help="fine-tune Chronos-2 on NHSN + ILINet history")
    t.add_argument("--cutoff", type=_date, default=dt.date.today(), help="exclude data on/after this date")
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--mode", choices=["lora", "full"], default="lora")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--prediction-length", type=int, default=8)
    t.add_argument("--no-ilinet", action="store_true", help="train on NHSN only")
    t.add_argument("--out", type=Path, help="checkpoint folder (default: models/chronos2-ft)")
    t.set_defaults(func=cmd_finetune)

    s = sub.add_parser("submit", help="open a pull request to the FluSight hub")
    s.add_argument("--reference-date", type=_date)
    s.add_argument("--out", type=Path, default=C.FORECAST_DIR)
    s.add_argument("--hub-dir", type=Path, default=os.environ.get("FLUSIGHT_HUB_DIR"))
    s.add_argument("--include-metadata", action="store_true", help="also submit model-metadata/<model>.yml")
    s.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    s.set_defaults(func=cmd_submit)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    changed = C.load_overrides(args.config)
    if changed:
        log.info("Settings from %s: %s", args.config, ", ".join(changed))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
