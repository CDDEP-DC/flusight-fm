"""How the hub's target data get revised, and which nowcast model handles it best.

`flusight revisions` downloads every committed version of the target data files
(the hub's git history), and:

1. tabulates, for every weekly data release, how it changed the weeks already reported;
2. measures how far each report is from its final value, by lag (weeks since the
   report) and season;
3. tests whether revision size is proportional to the count (linear), grows like
   its square root (count noise), or mixes the two, and whether it depends on the
   size or phase of the epidemic;
4. replays the nowcast week by week with each revision model in nowcast.py, using
   only the data versions available at the time, and scores it against final data.

Output: <root>/revisions/revisions.html (self-contained) and CSV tables.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config as C
from . import data
from . import nowcast as nc

log = logging.getLogger(__name__)
TEMPLATE = Path(__file__).with_name("revisions_template.html")
LABELS = {"hosp": "Hospital admissions", "ed": "ED visits (% flu)"}
SIZE_BINS = {"hosp": [0, 1, 10, 30, 100, 300, 1000, np.inf],
             "ed": [0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 1]}
GROWTH_BINS = [-np.inf, -0.3, -0.1, 0.1, 0.3, np.inf]
GROWTH_LABELS = ["drop > 26%", "drop 10-26%", "flat (±10%)", "rise 10-35%", "rise > 35%"]
FINAL_WEEKS = 8      # a week is scored against the latest data only once it is this many weeks old
MAX_LAG = 8
SCORE_LEVELS = np.array([0.01, 0.025] + [round(0.05 * i, 2) for i in range(1, 20)] + [0.975, 0.99])


def season_of(d) -> str:
    d = pd.Timestamp(d)
    y = d.year - (d.month < 8)
    return f"{y}-{(y + 1) % 100:02d}"


# ------------------------------------------------------------------ the data
def weekly_versions(key: str) -> tuple[dict[dt.date, pd.DataFrame], dict[dt.date, int]]:
    """The last version committed before each forecast deadline, and how many commits each week had."""
    versions = data.load_versions(key)
    weekly, commits = {}, {}
    for _, when, df in versions:
        ref = data.availability_date(str(when))
        weekly[ref] = df.dropna(subset=["value"]).drop_duplicates(["location", "date"], keep="last")
        commits[ref] = commits.get(ref, 0) + 1
    return dict(sorted(weekly.items())), commits


def weekly_changes(weekly: dict[dt.date, pd.DataFrame], commits: dict[dt.date, int], offset: float) -> pd.DataFrame:
    """One row per data release: what it reported for the new week, and how it revised earlier weeks."""
    refs = list(weekly)
    rows = []
    for prev_ref, ref in zip(refs, refs[1:]):
        old = weekly[prev_ref].set_index(["location", "date"])["value"]
        new = weekly[ref].set_index(["location", "date"])["value"]
        last, prev_last = new.index.get_level_values(1).max(), old.index.get_level_values(1).max()
        both = pd.concat([old, new], axis=1, keys=["old", "new"], join="inner")
        both["delta"] = both["new"] - both["old"]
        both["lag"] = ((last - both.index.get_level_values(1)).days // 7).to_numpy()
        states = both[both.index.get_level_values(0) != "US"]
        us = both.xs("US", level=0) if "US" in both.index.get_level_values(0) else pd.DataFrame(columns=both.columns)
        recent = us[us.index == prev_last]
        lag1 = states[states.index.get_level_values(1) == prev_last]
        big = lag1.loc[lag1["delta"].abs().idxmax()] if len(lag1) and lag1["delta"].abs().max() > 0 else None
        rows.append({
            "ref": ref, "commits": commits.get(ref, 1), "weeks_since_previous": (ref - prev_ref).days // 7,
            "newest_week": last.date(), "us_newest": new.get(("US", last), np.nan),
            "previous_newest_week": prev_last.date(),
            "us_first": recent["old"].iloc[0] if len(recent) else np.nan,
            "us_revised": recent["new"].iloc[0] if len(recent) else np.nan,
            "us_older_revision": us.loc[us.index < prev_last, "delta"].sum(),
            "locations_revised": int((lag1["delta"] != 0).sum()),
            "locations_revised_older": int(states.loc[states.index.get_level_values(1) < prev_last]
                                            .groupby(level=0)["delta"].apply(lambda d: (d != 0).any()).sum()),
            "largest_location": big.name[0] if big is not None else "",
            "largest_change": big["delta"] if big is not None else 0.0,
            "largest_change_pct": (big["delta"] / big["old"] if big is not None and big["old"] else np.nan),
        })
    out = pd.DataFrame(rows)
    out["us_change"] = out["us_revised"] - out["us_first"]
    out["us_change_pct"] = out["us_change"] / out["us_first"]
    return out


def report_vs_final(weekly, offset, lags=MAX_LAG + 1) -> pd.DataFrame:
    """Every report at lags 0..lags-1, against the latest data (weeks at least FINAL_WEEKS old)."""
    latest = weekly[max(weekly)]
    errs = nc.revision_errors(weekly, max(weekly), offset, lags=lags, truth=latest, settle=FINAL_WEEKS,
                              start=C.VERSIONS_START)
    errs = errs[errs["ref"] < max(weekly)]
    errs["season"] = errs["date"].map(season_of)
    errs["era"] = np.where(pd.to_datetime(errs["ref"]) >= pd.Timestamp(C.REVISION_START), "current", "earlier")
    return errs


def _intensity(weekly, rows: pd.DataFrame) -> np.ndarray:
    """Each report divided by its location's highest value in the previous 52 weeks of the same version."""
    out = np.full(len(rows), np.nan)
    for ref, idx in rows.groupby("ref").indices.items():
        wide = weekly[ref].pivot(index="date", columns="location", values="value")
        peak = wide.rolling(52, min_periods=8).max()
        r = rows.iloc[idx]
        p = peak.stack().reindex(pd.MultiIndex.from_arrays([r["date"], r["location"]])).to_numpy()
        out[idx] = r["rep"].to_numpy() / np.where(p > 0, p, np.nan)
    return out


# ------------------------------------------------------------------ analyses
def completeness(errs: pd.DataFrame) -> pd.DataFrame:
    """By season and lag: typical and aggregate report/final ratios, for states and the nation."""
    rows = []
    for (season, lag), g in errs.groupby(["season", "lag"]):
        st, us = g[g["location"] != "US"], g[g["location"] == "US"]
        if len(st) < 300:  # partial seasons at the edges of the data
            continue
        rows.append({"season": season, "lag": lag, "n": len(st),
                     "states_median_pct": 100 * (np.exp(st["err"].median()) - 1),
                     "states_aggregate_pct": 100 * (st["fin"].sum() / st["rep"].sum() - 1) if st["rep"].sum() else np.nan,
                     "states_revised_share": float((st["fin"] != st["rep"]).mean()),
                     "us_median_pct": 100 * (np.exp(us["err"].median()) - 1) if len(us) else np.nan,
                     "us_mean_pct": 100 * (np.exp(us["err"]).mean() - 1) if len(us) else np.nan})
    return pd.DataFrame(rows)


def size_bins(g: pd.DataFrame, key: str) -> pd.DataFrame:
    b = pd.cut(g["rep"], SIZE_BINS[key], right=False)
    rows = []
    for interval, gg in g.groupby(b, observed=True):
        rev = gg["fin"] - gg["rep"]
        lo, hi = np.quantile(gg["err"], [0.025, 0.975]) if len(gg) > 10 else (-np.inf, np.inf)
        rows.append({"size": _bin_label(interval, key), "n": len(gg),
                     "revised_share": float((rev != 0).mean()),
                     "median_pct": 100 * (np.exp(gg["err"].median()) - 1),
                     "aggregate_pct": 100 * (gg["fin"].sum() / gg["rep"].sum() - 1) if gg["rep"].sum() else np.nan,
                     "sd_log": float(np.clip(gg["err"], lo, hi).std()),
                     "median_abs_revision": float(rev.abs().median()),
                     "mean_abs_revision": float(rev.abs().mean())})
    return pd.DataFrame(rows)


def _bin_label(interval, key) -> str:
    lo, hi = interval.left, interval.right
    if key == "ed":
        return f"{lo * 100:g}–{hi * 100:g}%" if np.isfinite(hi) else f"≥ {lo * 100:g}%"
    if lo == 0 and hi == 1:
        return "0"
    return f"{lo:g}–{hi - 1:g}" if np.isfinite(hi) else f"≥ {lo:g}"


def power_fit(g: pd.DataFrame, n_boot: int = 300, seed: int = 1) -> dict:
    """log|revision| = a + b log(report), over reports that were revised; bootstrap CI over release weeks."""
    rev = (g["fin"] - g["rep"]).abs()
    ok = (g["rep"] > 0) & (rev > 0)
    x, y, refs = np.log(g.loc[ok, "rep"].to_numpy()), np.log(rev[ok].to_numpy()), g.loc[ok, "ref"].to_numpy()
    if len(x) < 20:
        return {}
    b, a = np.polyfit(x, y, 1)
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(x))).groupby(refs).apply(list).tolist()
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boots.append(np.polyfit(x[idx], y[idx], 1)[0])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"a": a, "b": b, "b_lo": lo, "b_hi": hi, "n": int(ok.sum())}


def variance_models(g: pd.DataFrame, key: str, offset: float) -> tuple[pd.DataFrame, dict]:
    """Compare forms for the revision variance on the log scale, fitted to the residuals of the
    size+growth mean model (y = report, u = offset / (y + offset)):

    - proportional:     var = s0^2                  -> |revision| grows linearly with the count
    - count noise:      var = s1^2 u                -> |revision| grows like sqrt(count)
    - mixed:            var = s0^2 + s1^2 u
    - power law:        var = c ((y + offset) / offset)^(-2k)  -> |revision| grows like count^(1-k)

    Returns the comparison table and, per form, the implied median |revision| over a range of sizes."""
    m = nc.fit_lag(g, "size+growth", offset)
    res = g["err"].to_numpy() - m.mean(g["rep"], g["growth"], g["location"])
    lo, hi = np.quantile(res, [0.025, 0.975])
    r2 = np.clip(res, lo, hi) ** 2
    y = g["rep"].to_numpy(float)
    u = offset / (y + offset)
    s_mix = nc._nnls2(np.column_stack([np.ones(len(g)), u]), r2)
    c, k = nc.fit_power_variance(y, r2, offset)
    forms = {
        "proportional (linear in count)": (lambda yy: np.full_like(yy, r2.mean()), 1, f"s.d. {np.sqrt(r2.mean()):.3f}"),
        "count noise (√count)": (lambda yy: (u @ r2 / (u @ u)) * offset / (yy + offset), 1,
                                 f"s.d. {np.sqrt(u @ r2 / (u @ u)):.3f} × √(offset / (report + offset))"),
        "proportional + count noise": (lambda yy: s_mix[0] + s_mix[1] * offset / (yy + offset), 2,
                                       f"s0 {np.sqrt(s_mix[0]):.3f}, s1 {np.sqrt(s_mix[1]):.3f}"),
        "power law": (lambda yy: c * ((yy + offset) / offset) ** (-2 * k), 2,
                      f"s.d. {np.sqrt(c):.3f} × size^−{k:.2f}, so |revision| ∝ count^{1 - k:.2f}"),
    }
    grid = np.unique(np.geomspace(max(y[y > 0].min(), offset), y.max(), 40))
    rows, curves = [], {}
    for name, (f, p, desc) in forms.items():
        ll = nc._gauss_loglik(r2, f(y))
        rows.append({"model": name, "params": p, "fit": desc, "loglik": ll, "aic": 2 * p - 2 * ll, "n": len(r2)})
        curves[name] = {"x": grid.tolist(), "y": (0.6745 * grid * np.sqrt(f(grid))).tolist()}
    out = pd.DataFrame(rows)
    out["delta_aic"] = out["aic"] - out["aic"].min()
    by_size = pd.DataFrame({"bin": pd.cut(y, SIZE_BINS[key], right=False), "r2": r2, "y": y})
    emp = by_size.groupby("bin", observed=True).agg(size=("y", "median"), sd=("r2", lambda v: float(np.sqrt(v.mean()))),
                                                     n=("y", "size"))
    for name, (f, _, _) in forms.items():
        emp[name] = np.sqrt(f(emp["size"].to_numpy(float)))
    emp.index = [_bin_label(b, key) for b in emp.index]
    return out, {"curves": curves, "power_k": k, "by_size": emp.reset_index(names="bin")}


def binned_effect(g: pd.DataFrame, by: pd.Series, labels=None) -> pd.DataFrame:
    rows = []
    for k, gg in g.groupby(by, observed=True):
        lo, hi = np.quantile(gg["err"], [0.025, 0.975]) if len(gg) > 10 else (-np.inf, np.inf)
        rows.append({"bin": str(k), "n": len(gg), "median_pct": 100 * (np.exp(gg["err"].median()) - 1),
                     "mean_pct": 100 * (np.exp(gg["err"]).mean() - 1),
                     "sd_log": float(np.clip(gg["err"], lo, hi).std())})
    out = pd.DataFrame(rows)
    if labels is not None and len(out) == len(labels):
        out["bin"] = labels
    return out


def national_size(errs: pd.DataFrame) -> dict:
    """Is the nation's newest-week revision (as a share) related to how big the epidemic is?"""
    us = errs[(errs["location"] == "US") & (errs["lag"] == 0) & (errs["era"] == "current")]
    if len(us) < 10:
        return {}
    x = np.log(us["rep"].to_numpy())
    beta = nc._huber(np.column_stack([np.ones(len(x)), x - x.mean()]), us["err"].to_numpy())
    rng = np.random.default_rng(2)
    boots = [nc._huber(np.column_stack([np.ones(len(i)), x[i] - x[i].mean()]), us["err"].to_numpy()[i])[1]
             for i in (rng.integers(0, len(x), len(x)) for _ in range(300))]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"slope": float(beta[1]), "lo": float(lo), "hi": float(hi), "n": len(us),
            "points": {"date": [str(d.date()) for d in us["date"]], "rep": us["rep"].round(1).tolist(),
                       "pct": (100 * (np.exp(us["err"]) - 1)).round(2).tolist()}}


# ------------------------------------------------------------ nowcast replay
_erf = np.vectorize(math.erf)


def _crps_normal(mu, sd, y):
    z = (y - mu) / sd
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + _erf(z / math.sqrt(2)))
    return sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def _norm_ppf(p):
    return np.array([NormalDist().inv_cdf(float(q)) for q in np.atleast_1d(p)])


# (label, mean model, variance form) compared in the replay
VARIANTS = [("none", "none", "power"),
            ("constant (previous model)", "constant", "constant"),
            ("size", "size", "power"),
            ("size+growth, constant s.d.", "size+growth", "constant"),
            ("size+growth", "size+growth", "power")]
PREVIOUS = "constant (previous model)"


def in_use_label(key: str) -> str:
    kind = C.NOWCAST_MODEL.get(key, "constant")
    return next((label for label, k, v in VARIANTS if k == kind and v == C.NOWCAST_VARIANCE), kind)


def replay(weekly, offset, variants=VARIANTS, lags: int | None = None, min_train: int = 200) -> pd.DataFrame:
    """For each past data release, fit each revision model on the versions available then, nowcast that
    release's latest weeks, and score against the latest data. One row per (release, location, lag, model)."""
    lags = C.NOWCAST_WEEKS if lags is None else lags
    latest = weekly[max(weekly)]
    cutoff = latest["date"].max() - pd.Timedelta(weeks=FINAL_WEEKS)
    zq = _norm_ppf(SCORE_LEVELS)
    rows = []
    for ref in weekly:
        if ref < pd.Timestamp(C.REVISION_START).date() or weekly[ref]["date"].max() > cutoff:
            continue
        train = nc.revision_errors(weekly, ref, offset)
        if len(train) < min_train:
            continue
        test = nc.revision_errors({ref: weekly[ref]}, ref, offset, lags=lags, truth=latest, settle=FINAL_WEEKS,
                                  start=str(ref))
        if test.empty:
            continue
        y, z, fin = test["err"].to_numpy(), np.log(test["rep"].to_numpy() + offset), test["fin"].to_numpy()
        for label, kind, variance in variants:
            mu, sd = np.zeros(len(test)), np.full(len(test), C.NOWCAST_FALLBACK_SD)
            for lag, idx in test.groupby("lag").indices.items():
                g = train[train["lag"] == lag]
                if len(g) < 20:
                    continue
                m = nc.fit_lag(g, kind, offset, variance)
                t = test.iloc[idx]
                mu[idx] = m.mean(t["rep"], t["growth"], t["location"])
                sd[idx] = m.sd(t["rep"])
            q = np.maximum(np.exp(z[:, None] + mu[:, None] + sd[:, None] * zq[None, :]) - offset, 0)
            pin = (SCORE_LEVELS[None, :] - (fin[:, None] < q)) * (fin[:, None] - q)
            rows.append(pd.DataFrame({
                "ref": ref, "location": test["location"].to_numpy(), "lag": test["lag"].to_numpy(), "model": label,
                "abs_err_log": np.abs(y - mu), "crps_log": _crps_normal(mu, sd, y), "wis": 2 * pin.mean(axis=1),
                "in50": np.abs(y - mu) <= 0.6745 * sd, "in90": np.abs(y - mu) <= 1.6449 * sd,
                "abs_err_count": np.abs(np.maximum(np.exp(z + mu) - offset, 0) - fin)}))
        rows.append(pd.DataFrame({"ref": ref, "location": test["location"].to_numpy(), "lag": test["lag"].to_numpy(),
                                  "model": "report as-is", "abs_err_log": np.abs(y),
                                  "abs_err_count": np.abs(test["rep"].to_numpy() - fin)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def replay_summary(rep: pd.DataFrame) -> pd.DataFrame:
    if rep.empty:
        return rep
    rep = rep.assign(season=rep["ref"].map(season_of))
    agg = dict(n=("abs_err_log", "size"), abs_err_log=("abs_err_log", "mean"), crps_log=("crps_log", "mean"),
               wis=("wis", "mean"), cov50=("in50", "mean"), cov90=("in90", "mean"))
    us = rep[rep["location"] == "US"].groupby(["lag", "model"])["abs_err_count"].mean().rename("us_abs_err")
    out = rep.groupby(["lag", "model"]).agg(**agg).join(us).reset_index()
    by_season = rep.groupby(["season", "lag", "model"]).agg(**agg).reset_index()
    return pd.concat([out.assign(season="all"), by_season], ignore_index=True)


def fitted_models(weekly, offset, kind) -> pd.DataFrame:
    models = nc.fit_revisions(weekly, max(weekly), offset, kind)
    sizes = [1, 10, 100, 1000, 10000] if offset >= 1 else [0.002, 0.005, 0.01, 0.02, 0.05]
    rows = []
    for lag, m in models.items():
        for s in sizes:
            for g, label in ((0.0, "flat"), (-0.3, "drop 26%"), (0.3, "rise 35%")):
                rows.append({"lag": lag, "model": kind, "report": s, "trend": label,
                             "median_shift_pct": 100 * (np.exp(float(m.mean(s, g)[0])) - 1),
                             "sd_log": float(m.sd(s)[0]), "n": m.n})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ assembly
def _clean(o, digits: int = 4):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> Python, dates -> ISO strings, floats rounded."""
    if isinstance(o, dict):
        return {str(k): _clean(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v, digits) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float):
        return round(o, digits) if math.isfinite(o) else None
    if isinstance(o, (dt.date, pd.Timestamp)):
        return str(o)[:10]
    return o


def _records(df: pd.DataFrame, digits: int = 4) -> list[dict]:
    return _clean(df.to_dict(orient="records"), digits)


def _findings(key: str, d: dict) -> list[str]:
    out = []
    comp = pd.DataFrame(d["completeness"])
    full = comp[(comp["lag"] == 0) & (comp["n"] >= 300)]["season"] if len(comp) else []
    cur = comp[comp["season"] == max(full)] if len(full) else comp.iloc[:0]
    if len(cur):
        l0 = cur[cur["lag"] == 0].iloc[0] if (cur["lag"] == 0).any() else None
        if l0 is not None:
            out.append(f"In {l0['season']}, the newest week's first report ended up {l0['states_median_pct']:+.1f}% "
                       f"(median state) and {l0['us_median_pct']:+.1f}% (nation) from its final value; "
                       f"{100 * l0['states_revised_share']:.0f}% of state reports were revised at all.")
        late = cur[cur["lag"] >= 2]
        if len(late) and late["us_median_pct"].notna().any():
            if late["us_median_pct"].abs().max() >= 0.5:
                out.append(f"Revisions continue after two weeks: nationally, weeks 2-{int(late['lag'].max())} old are still "
                           f"{late['us_median_pct'].min():+.1f}% to {late['us_median_pct'].max():+.1f}% from final. These "
                           f"come from a minority of states; the median state's report is not revised further.")
            else:
                out.append(f"Reports are essentially final after two weeks: nationally within "
                           f"{late['us_median_pct'].abs().max():.1f}% of the final value.")
    vm = pd.DataFrame(d["variance_models"])
    k = (d.get("variance_detail") or {}).get("power_k")
    if len(vm):
        best = vm.loc[vm["aic"].idxmin()]
        others = ", ".join(f"'{r['model']}' by {r['delta_aic']:.0f}" for _, r in vm.iterrows() if r["model"] != best["model"])
        if best["model"] == "power law" and k is not None:
            out.append(f"Revisions are not linear in the count. Their spread grows like count^{1 - k:.2f}: less than "
                       f"proportionally, so relative revisions shrink steadily as reports get bigger (a 10x larger report has "
                       f"{100 * (1 - 10 ** -k):.0f}% smaller relative revisions). This power law fits better than the "
                       f"alternatives (AIC worse for {others}).")
        else:
            out.append(f"Of the forms for revision size, '{best['model']}' fits best (AIC worse for {others}).")
    pw = d.get("power") or {}
    if pw:
        out.append(f"Among reports that were revised, |revision| grows as report^{pw['b']:.2f} (95% CI {pw['b_lo']:.2f}-"
                   f"{pw['b_hi']:.2f}); small reports are both revised less often and by a larger share when they are.")
    ns = d.get("national_size") or {}
    if ns:
        sig = not (ns["lo"] <= 0 <= ns["hi"])
        out.append(f"Nationally, the newest week's revision as a share of the report {'does' if sig else 'does not clearly'} depend "
                   f"on the size of the epidemic: slope {ns['slope']:+.3f} per log unit of the national value (95% CI "
                   f"{ns['lo']:+.3f} to {ns['hi']:+.3f})" + ("." if sig else ", so at the national level revisions are "
                                                            "roughly proportional (linear) to the count."))
    gr = pd.DataFrame(d["growth"])
    if len(gr) >= 2:
        out.append(f"Epidemic phase matters: when the newest report shows a {gr.iloc[0]['bin']}, it is later revised by "
                   f"{gr.iloc[0]['median_pct']:+.1f}% (median), vs {gr.iloc[-1]['median_pct']:+.1f}% when it shows a {gr.iloc[-1]['bin']}.")
    rs = pd.DataFrame(d["replay"])
    chosen = d["model"]
    if len(rs):
        for lag in (0, 1):
            a = rs[(rs["season"] == "all") & (rs["lag"] == lag)].set_index("model")
            if PREVIOUS not in a.index or chosen not in a.index or not a.at[PREVIOUS, "wis"]:
                continue
            change = 100 * (a.at[chosen, "wis"] / a.at[PREVIOUS, "wis"] - 1)
            week = "newest week" if lag == 0 else "week before it"
            if chosen.startswith("none"):
                out.append(f"Replaying the nowcast week by week ({week}): not shifting the report changes WIS by {change:+.0f}% "
                           "vs the previous fixed-shift model, so for this dataset the report is kept and only its revision "
                           "uncertainty is applied.")
            else:
                us = (f"; the national nowcast error is {a.at[chosen, 'us_abs_err']:,.0f} vs {a.at[PREVIOUS, 'us_abs_err']:,.0f} "
                      f"before and {a.at['report as-is', 'us_abs_err']:,.0f} for the report as-is"
                      if pd.notna(a.at[chosen, "us_abs_err"]) else "")
                out.append(f"Replaying the nowcast week by week ({week}), '{chosen}' (now used) changes WIS by {change:+.0f}% vs "
                           f"the previous fixed-shift model{us}.")
    return out


def analyze(key: str) -> dict:
    offset = C.LOG_OFFSET[key]
    weekly, commits = weekly_versions(key)
    log.info("%s: %d weekly versions from %d commits", key, len(weekly), sum(commits.values()))
    changes = weekly_changes(weekly, commits, offset)
    errs = report_vs_final(weekly, offset)
    cur0 = errs[(errs["lag"] == 0) & (errs["location"] != "US") & (errs["era"] == "current")].copy()
    cur0["intensity"] = _intensity(weekly, cur0)

    locs = data.load_locations()
    names = dict(zip(locs["location"], locs["location_name"]))
    latest = weekly[max(weekly)]
    traj = {}
    for loc, g in errs[errs["lag"] <= 1].groupby("location"):
        w = g.pivot_table(index="date", columns="lag", values="rep")
        fin = latest[latest["location"] == loc].set_index("date")["value"]
        w = w.join(fin.rename("final"), how="outer").sort_index()
        w = w[w.index >= errs["date"].min()]
        traj[loc] = {"dates": [str(x.date()) for x in w.index],
                     "first": [None if pd.isna(v) else float(v) for v in w.get(0, pd.Series(np.nan, index=w.index))],
                     "second": [None if pd.isna(v) else float(v) for v in w.get(1, pd.Series(np.nan, index=w.index))],
                     "final": [None if pd.isna(v) else float(v) for v in w["final"]]}

    rng = np.random.default_rng(0)
    pts = cur0[(cur0["rep"] > 0) & (cur0["fin"] != cur0["rep"])]
    pts = pts.iloc[rng.permutation(len(pts))[:2500]]
    scatter = {"rep": pts["rep"].round(5).tolist(), "rev": (pts["fin"] - pts["rep"]).abs().round(5).tolist()}

    log.info("%s: replaying the nowcast with each revision model", key)
    rep = replay(weekly, offset)
    var_table, var_detail = variance_models(cur0, key, offset)
    out = {
        "label": LABELS[key], "unit": "count" if key == "hosp" else "proportion", "offset": offset,
        "versions": int(sum(commits.values())), "weeks": len(weekly),
        "first_version": str(min(weekly)), "last_version": str(max(weekly)),
        "locations": [{"code": c, "name": names.get(c, c)} for c in sorted(traj, key=lambda c: (c != "US", names.get(c, c)))],
        "weekly": _records(changes, 5),
        "trajectories": traj,
        "completeness": _records(completeness(errs)),
        "size_bins": _records(size_bins(cur0, key), 5),
        "power": _clean(power_fit(cur0)),
        "variance_models": _records(var_table),
        "variance_detail": {"curves": _clean(var_detail["curves"], 6), "power_k": var_detail["power_k"],
                            "by_size": _records(var_detail["by_size"])},
        "scatter": scatter,
        "growth": _records(binned_effect(cur0, pd.cut(cur0["growth"], GROWTH_BINS), GROWTH_LABELS)),
        "intensity": _records(binned_effect(cur0, pd.cut(cur0["intensity"], [0, 0.1, 0.25, 0.5, 0.75, 1.01]),
                                            ["< 10% of peak", "10-25%", "25-50%", "50-75%", "75-100%"])),
        "national_size": _clean(national_size(errs)),
        "replay": _records(replay_summary(rep)),
        "fitted": _records(fitted_models(weekly, offset, C.NOWCAST_MODEL.get(key, "constant"))),
        "model": in_use_label(key),
        "variance": C.NOWCAST_VARIANCE,
    }
    out["findings"] = _findings(key, out)
    out["_tables"] = {"weekly_changes": changes, "report_vs_final": errs, "replay_rows": rep}
    return out


def run(root: Path, keys=("hosp", "ed")) -> Path:
    root = Path(root) / "revisions"
    root.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "hub": C.HUB_REPO,
                        "settle_weeks": C.REVISION_SETTLE_WEEKS, "final_weeks": FINAL_WEEKS,
                        "revision_start": C.REVISION_START, "nowcast_weeks": C.NOWCAST_WEEKS},
               "datasets": {}}
    for key in keys:
        d = analyze(key)
        for name, df in d.pop("_tables").items():
            df.to_csv(root / f"{key}-{name}.csv", index=False)
        for name in ("completeness", "size_bins", "variance_models", "growth", "intensity", "replay", "fitted"):
            pd.DataFrame(d[name]).to_csv(root / f"{key}-{name}.csv", index=False)
        payload["datasets"][key] = d
    blob = json.dumps(_clean(payload, 6), allow_nan=False)
    out = root / "revisions.html"
    out.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", blob.replace("</", "<\\/")), encoding="utf-8")
    log.info("Revision report: %s", out)
    return out
