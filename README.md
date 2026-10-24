# OHTJHU-BM: FluSight foundation-model ensemble

Weekly forecasts for the [FluSight 2026-27 hub](https://github.com/cdcepi/FluSight-forecast-hub).
They come from a zero-shot ensemble of Chronos-2, TimesFM 2.5, Moirai 2.0 and Timestral,
with a backfill nowcast, a pre-submission review report, backtesting and optional
fine-tuning. All models run locally from open weights; no forecasting API keys are needed.

## Weekly workflow

```bash
flusight forecast     # nowcast + models + ensemble -> submission file + review report
                      # open the report link it prints and check it
flusight submit       # opens a pull request to the hub (asks for confirmation)
```

Submissions are due **Wednesdays by 11 PM ET**. The hub's target data updates
Wednesday around midday, so run after that.

## What it produces

| target | horizons | output |
|---|---|---|
| `wk inc flu hosp` | -1 to 3 | 23 quantiles (integers) |
| `wk flu hosp rate change` | 0 to 3 | pmf over 5 categories |
| `wk inc flu prop ed visits` | -1 to 3 | 23 quantiles (proportions) |
| `peak week inc flu hosp` | blank | pmf over season Saturdays |
| `peak inc flu hosp` | blank | 23 quantiles (integers) |

### How a forecast is built

1. **Data.** Download the hub's target data, plus past versions of it rebuilt
   from the hub's git history (cached under `cache/`).
2. **Nowcast** (`nowcast.py`). The last 2 reported weeks are usually revised
   upward later. Each is estimated from two sources:
   - the report, adjusted by a revision model learned from every past version of
     the hub's data. For hospital admissions, the expected revision depends on the
     report's size and on its week-over-week change: a sharp apparent drop is
     usually an incomplete report. The uncertainty shrinks as a power of the
     report's size. ED visit reports are kept as they are, with uncertainty only.
   - last week's forecast for that week.

   The models then see the nowcast-adjusted series, and horizon -1 is submitted
   from the nowcast distribution. `flusight revisions` produces the analysis
   behind these choices (see below).
3. **Models** (`models.py`). Each model forecasts every location. A model that
   fails is logged and left out.
4. **Ensemble** (`forecast.py`). Quantiles are mapped onto FluSight's 23 levels,
   then combined with a median (or mean, or weighted) across models.
5. **Derived targets.** Rate-change probabilities come from the ensemble's
   quantiles. Peak targets come from 2,000 sampled season trajectories.
6. **Outputs**, under `forecasts/`:
   - `model-output/` holds the submission file.
   - `components/` holds each model's own forecasts, for evaluation and tuning.
   - `nowcasts/` holds the nowcast tables.
   - `reports/` holds the review page.

### The review report

`forecasts/reports/<date>-report.html` is a single file that works offline. It shows:
- **Checks**: validation errors, failed models, large nowcast revisions, big
  jumps from last week's forecast, and models that disagree.
- **An explorer for each location**: reported data, nowcast, ensemble fan,
  each model's median (toggle in the legend), and last week's forecast.
- **Rate-change probabilities** and the **peak-week distribution**.
- **Small multiples** of all locations, and a **summary table**.
- **Evaluation** of past submissions against current data: relative WIS
  against FluSight-baseline, and coverage.

## Setup (any machine)

The code runs on **macOS (Apple Silicon), Windows (x64) and Linux**. Everything runs on CPU;
a CUDA GPU is used automatically if present. [uv](https://docs.astral.sh/uv/) manages Python
the same way on all three.

**Install uv, git and the GitHub CLI**

| | |
|---|---|
| macOS | `brew install uv git gh` (or `curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| Windows | `winget install astral-sh.uv Git.Git GitHub.cli` |
| Linux | `curl -LsSf https://astral.sh/uv/install.sh \| sh`, plus git and gh from your package manager |

**Clone and install.** Keep the clone *outside* OneDrive/iCloud/Dropbox: syncing a
virtual environment and caches is slow and can corrupt them.

```bash
git clone https://github.com/<you>/flusight-fm.git ~/code/flusight-fm
cd ~/code/flusight-fm
uv venv --python 3.11
# activate: source .venv/bin/activate (macOS/Linux)  |  .venv\Scripts\Activate.ps1 (Windows)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux/Windows; on macOS just: uv pip install torch
uv pip install -e ".[all]"
flusight --help
```

**Using Anaconda/conda instead of uv** (any OS):

```bash
conda create -n flusight -c conda-forge python=3.11 git gh -y
conda activate flusight
pip install torch --index-url https://download.pytorch.org/whl/cpu   # macOS: pip install torch
pip install -e ".[all]"
```

`scripts/weekly.ps1` finds the conda env `flusight` automatically if there is no `.venv`.

On a Linux GPU machine, install the default CUDA torch wheel instead of the CPU one.
To fine-tune, also install `uv pip install -e ".[finetune]"`.

**Keys and paths.** Set these in your shell profile (macOS/Linux) or with `setx` (Windows):

| variable | needed for |
|---|---|
| `FLUSIGHT_HUB_DIR` | `flusight submit`: path to your clone of your fork of the hub |
| `GITHUB_TOKEN` | optional; raises GitHub API limits when downloading data versions |
| `DELPHI_API_KEY` | optional; for `fetch-history` (ILINet) |
| `FLUSIGHT_DEVICE` | optional: `cpu`, `cuda`, or `mps` (Apple GPU, experimental) |

**Hub fork** (once, on each machine you submit from):

```bash
gh auth login
gh repo fork cdcepi/FluSight-forecast-hub --clone=false
git clone https://github.com/<you>/FluSight-forecast-hub ~/code/FluSight-forecast-hub
git -C ~/code/FluSight-forecast-hub remote add upstream https://github.com/cdcepi/FluSight-forecast-hub
```

**Register the model**: fill in the `TODO`s in `model-metadata/OHTJHU-BM.yml`, then run
`flusight submit --include-metadata` the first time (or open a separate PR with just that file).

### Platform notes

- **Mac Mini (Apple Silicon)**: supported. All models run on CPU.
  `FLUSIGHT_DEVICE=mps` may speed things up but is untested with every model.
  Intel Macs are **not** supported, because current PyTorch no longer ships Intel macOS builds.
- **Windows on ARM** (Snapdragon laptops): PyTorch and several model packages lack native
  ARM64 Windows builds. Use WSL2 (Ubuntu), which follows the Linux instructions.
- **Linux / cloud**: supported. The `Dockerfile` builds a CPU image for batch jobs.
- **First run** on a machine downloads about 1-2 GB of model weights to `~/.cache/huggingface`.

## Running it automatically

- **GitHub Actions** (`.github/workflows/weekly.yml`), the most reliable option.
  - Runs `flusight forecast` every Wednesday at 19:00 UTC on GitHub's Linux runners.
  - Commits `forecasts/` back to the repo and uploads the report as a downloadable artifact.
  - Doesn't submit. Pull, review the report, then run `flusight submit` locally.
  - A private repo gets 2,000 free runner minutes a month; this uses roughly 30-60 per week.
- **macOS/Linux cron**:
  `0 14 * * 3 bash ~/code/flusight-fm/scripts/weekly.sh`
  On macOS, give the terminal Full Disk Access, or use a launchd agent.
- **Windows Task Scheduler**:
  `schtasks /Create /TN FluSight /SC WEEKLY /D WED /ST 14:00 /TR "powershell -File C:\path\to\flusight-fm\scripts\weekly.ps1"`

`scripts/weekly.*` pull the repo, build the forecast and write a log under `logs/`. Pass
`--submit` (`-Submit` on Windows) to also open the PR without confirmation.

## Testing on last season

```bash
flusight season-test                 # all of last season (2025-26), every week
flusight season-test --every 2 --models chronos2 timesfm moirai2    # quicker first pass
```

This replays each week of the season. It uses only the data that was available at that
week's Wednesday deadline: the hub's data as committed that week, revision history up to then,
and its own earlier forecasts as the nowcast prior. Each week is then scored against today's
data. The weekly results table goes to
`backtests/season-2025-26/evaluation/weekly_results.html` (also `.csv` and `.md`), with one row
per week plus a season total:
- WIS and relative WIS against FluSight-baseline
- interval coverage
- nowcast error vs the raw report
- each model's own score
- the national forecast vs what happened

The test resumes where it left off if interrupted. A full season is roughly 30 weekly runs,
so expect a few hours on CPU.

Pretraining caveat: the foundation models' training data may overlap the test season. That
would make zero-shot results look better than they would have in real time, especially for
models released after the season (Timestral, Oct 2026).

## How the data get revised

```bash
flusight revisions        # -> forecasts/revisions/revisions.html (+ CSV tables)
```

This downloads every committed version of the hub's target data files (since
2023) and reports:
- each weekly release's revisions to earlier weeks, nationally and by state
- how far reports are from final, by lag and season
- whether revision size is linear in the count (it isn't: it grows like
  count^0.8 for hospital admissions)
- how revisions depend on epidemic size and phase
- a week-by-week replay of the nowcast with each revision model, scored
  against final data

The nowcast settings in `config.py` (`NOWCAST_MODEL`, `NOWCAST_VARIANCE`) were
chosen from that replay. Re-run it each season, or when NHSN reporting changes.

## Other commands

```bash
flusight report                    # re-render this week's report with fresh scores
flusight evaluate                  # score past submissions vs current data -> forecasts/evaluation/
flusight backtest --start 2024-11-23 --end 2025-05-31 --name zs-2425
flusight evaluate --dir backtests/zs-2425 --combinations
flusight fetch-history             # NHSN since 2020 + ILINet since 1997
flusight finetune --cutoff 2024-10-01
```

See **TUNING.md** for the tuning plan. Settings live in `flusight_fm/config.py`; override them in
`flusight.toml` at the repo root (each experiment is then a small committed file).

## Public domain

This repository's code, forecasts and documentation are dedicated to the **public domain** under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) (see `LICENSE`). You can copy,
modify, distribute and use them for any purpose, commercial or not, without asking permission or
giving attribution. The forecasts submitted to FluSight carry the same `CC0-1.0` designation in
`model-metadata/OHTJHU-BM.yml`. Third-party models and data used by the pipeline keep their own terms.

## Repository layout

```
flusight_fm/         the package (cli, data, nowcast, models, forecast, evaluate, report, backtest, finetune, submit)
forecasts/           weekly outputs (tracked, so every machine sees the history)
backtests/<name>/    backtest outputs (only evaluation/ is tracked)
model-metadata/      hub metadata file
scripts/             weekly run scripts for macOS/Linux and Windows
cache/, models/      downloads and fine-tuned checkpoints (not tracked)
```
