# Tuning plan for OHTJHU-BM

This plan covers what to tune, in what order, how to measure it, and how to decide.
Every knob named here is a setting in `flusight_fm/config.py`. Override settings in
`flusight.toml` rather than editing the code, so each experiment is a small file you
can keep with its results.

## Ground rules

1. **Out-of-sample only.** A setting is judged only on forecasts made with data
   available at the time. `flusight backtest` enforces this. Each week uses the
   hub data as it stood then (rebuilt from the hub's git history), revision
   history up to then, and its own earlier forecasts as the nowcast prior.
   Fine-tuned models need a training `--cutoff` before the season being scored.
2. **Tune on one season, confirm on another.**
   - Tuning season: **2024-25** (reference dates 2024-11-23 to 2025-05-31).
   - Confirmation season: **2025-26** (2025-11-22 to 2026-05-30).

   A change is adopted only if it helps on the tuning season *and* doesn't hurt
   on the confirmation season. Don't look at 2025-26 results while choosing.
3. **Primary metric: relative WIS on the log scale** (`rel_wis_log`), against
   `FluSight-baseline`, averaged over horizons 0-3 and all locations. The log
   scale stops the national and largest states from dominating, and FluSight's
   own evaluations increasingly use it. Also check:
   - relative WIS on the natural scale (`rel_wis`), which is dominated by large locations
   - 50% and 95% coverage, which should be near 50% and 95%
   - the Brier score for rate change
   - results by horizon and by season phase (growth, peak, decline)
4. **Effective sample size is small.** Two seasons of strongly correlated
   locations is closer to "two experiments" than to thousands. Prefer coarse
   choices (on/off, a few grid values) over fine continuous optimization. Treat
   improvements under about 3% in relative WIS as noise unless they hold in both
   seasons and at most horizons.
5. **Change things in season rarely.** Freeze the configuration before the
   season starts. Any mid-season change gets a dated note in the metadata's
   `methods_long` and in the results log below.

## Step 0: reference backtests (do first)

```bash
flusight fetch-history
flusight backtest --start 2024-11-23 --end 2025-05-31 --name zs-2425
flusight backtest --start 2025-11-22 --end 2026-05-30 --name zs-2526   # run now, but don't look until step 6
```

These zero-shot runs, with all five models and default settings, are the
reference every later experiment is compared with. Each week's component
forecasts are saved, so steps 1-2 need **no model reruns**.

Cost: one run is about 5 models x 53 locations x ~28 weeks. On CPU, budget a few
hours per season. Use a cloud Linux machine and `--every 2` for quick passes.
TimeGPT calls cost API credits, so use `--models` to leave it out of exploratory runs.

Before starting, check how far back the hub's history goes. Data versions are
only rebuilt from `revision_start` (2024-11-01) onward. The first weeks of
2024-25 coincide with NHSN reporting becoming mandatory again, so look at the
data around then before trusting early-season scores.

## Step 1: ensemble composition (cheap: no reruns)

```bash
flusight evaluate --dir backtests/zs-2425 --combinations
```

This scores every subset of two or more models under median and mean ensembling.

| Knob | Values to try | Notes |
|---|---|---|
| `models` | subsets from the combinations table | Drop a model only if removing it helps in most horizons. |
| `ensemble_method` | `median`, `mean` | With 3-5 members, median is usually more robust. |
| `ensemble_method = "weighted"` + `ensemble_weights` | inverse of each model's `rel_wis_log` on the tuning season | Only if it beats the best unweighted subset by more than 3%. |

Watch for models that are good on average but miss turning points. The
`by_week` table shows this.

## Step 2: calibration (cheap: no reruns)

Look at `cov50` and `cov95` by horizon for the chosen ensemble in
`backtests/zs-2425/evaluation/by_horizon.csv`.

| Knob | Values | Rule |
|---|---|---|
| `spread_scale` (per horizon 0-3) | 0.8, 0.9, 1.0, 1.15, 1.3 | Pick the value that brings 95% coverage closest to 0.95 without raising WIS. |

`spread_scale` is applied after ensembling, so it needs a re-run of the
ensemble stage only. Today that means rerunning the backtest. Adding a
`flusight evaluate --spread` option that reuses the saved components is a
worthwhile next step.

## Step 3: nowcast (cheap: no model runs needed to judge the nowcast itself)

The nowcast affects two things: the horizon -1 submission, and the inputs to
every model. Judge it directly first.

- **Direct accuracy.** Compare `backtests/<run>/nowcasts/*-hosp.csv` with the
  current data. Columns are `median` (revised) and `reported`; the current data
  is the "truth". Report the mean absolute log error of `median` vs `reported`.
  A nowcast that doesn't beat "use the report as-is" should be turned off.
- **Horizon -1 WIS** in the `by_horizon` table.

| Knob | Values | Notes |
|---|---|---|
| `nowcast` | true / false | false = the report as-is, with a fixed band |
| `nowcast_weeks` | 1, 2, 3 | How many recent weeks get revised |
| `prior_weight` | 0, 0.5, 1 | Weight on our previous forecast |
| `revision_shrink` | 2, 5, 20 | Low = trust each location's own revision history |
| `revision_start` | 2024-11-01, 2025-08-01 | Use only the current reporting regime? |

Then confirm the downstream effect with one rerun of the best nowcast setting:
horizon 0-3 WIS should not get worse.

## Step 4: preprocessing (needs reruns)

| Knob | Values | Notes |
|---|---|---|
| `transform` | `none`, `sqrt`, `log1p` | Foundation models scale internally. log1p can help small states and growth phases. |
| `context_start` | none, `2022-08-01`, `2023-08-01` | Early NHSN data (2020-22) reflects COVID-era flu dynamics and voluntary reporting. |

Run each as a separate backtest with the step 1-3 choices fixed. Use `--every 2`
for a first pass and confirm with the full run.

## Step 5: fine-tuning (expensive; last)

The models are zero-shot by default. Fine-tuning *may* help, because flu has
a very regular seasonal shape. It can also make things worse: overfitting, or
losing the model's general skill. Only the backtest can tell.

```bash
# Train only on data before the tuning season, then score that season.
flusight finetune --cutoff 2024-10-01 --out models/chronos2-ft-2410
# flusight.toml:  chronos2_finetuned = "models/chronos2-ft-2410/finetuned-ckpt"
flusight backtest --start 2024-11-23 --end 2025-05-31 --name ft-2425 --models chronos2 chronos2_ft timesfm moirai2 timestral
```

| Knob | Values | Notes |
|---|---|---|
| `--mode` | `lora`, `full` | LoRA is cheaper and less prone to forgetting. Start there. |
| `--steps` | 500, 1000, 3000 | |
| `--lr` | 1e-5, 3e-5 (LoRA); 1e-6 (full) | |
| `--no-ilinet` | with / without ILINet | ILINet adds 25+ seasons of epidemic shapes, but it is a different signal. |
| `--prediction-length` | 4, 8 | |
| `timegpt_finetune_steps` (model `timegpt_ft`) | 0, 10, 50 | TimeGPT fine-tunes per request on each series' own history. |

Compare `chronos2_ft` against `chronos2` as components, and the ensemble with
each. Hardware: LoRA fine-tuning on CPU is slow but possible. A cloud GPU, even
a small one, makes a 1000-step run take minutes.

Other models:
- **TimesFM 2.5** and **Moirai 2.0** can be fine-tuned with their own repos'
  training code. Add them only if Chronos-2 fine-tuning clearly helps.
- **Timestral** fine-tunes through Mistral's hosted service, which isn't wired in.

## Step 6: confirm and freeze

1. Write the chosen settings to `flusight.toml` and commit it.
2. Backtest 2025-26 with them (`--name final-2526`) and compare with `zs-2526`.
   Adopt only if it isn't worse.
3. Update `model-metadata/OHTJHU-BM.yml` (`methods`, `methods_long`, `model_version`).

## In-season monitoring (weekly, from the review report)

- If the trailing 4-week relative WIS of our ensemble is above 1.0 (worse than
  the baseline) for 4 or more consecutive weeks, look at which component is
  responsible before changing anything.
- If 95% coverage over the trailing 6 weeks drops below 80%, consider raising
  `spread_scale` for the affected horizons, and log it.
- Peak targets: score at season end. Then tune `peak_rho` (0.8, 0.9, 0.95) on
  the backtests' peak forecasts. That requires `--targets hosp rate_change peak`
  in the backtest.

## Results log

| Date | Experiment (`--name`) | Settings changed | rel WIS log (tuning) | rel WIS log (confirm) | Decision |
|---|---|---|---|---|---|
| | zs-2425 | defaults | | | reference |
