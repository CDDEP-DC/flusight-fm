"""Fine-tuning on long flu histories.

Only Chronos-2 is fine-tuned here (`Chronos2Pipeline.fit`). See TUNING.md for
the other models.

The training corpus is:
- NHSN weekly flu admissions by jurisdiction since Aug 2020 (data.cdc.gov), and
- optionally ILINet % ILI for the nation, HHS regions and states since 1997
  (Delphi). It is a different signal, but has 25+ seasons of epidemic shapes.

Everything after `cutoff` is excluded, so a model fine-tuned with a cutoff
before a backtest season can be evaluated fairly on that season.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import data
from .models import FREQ, device

log = logging.getLogger(__name__)
MIN_SEGMENT_WEEKS = 52


def _segments(df: pd.DataFrame, cutoff: dt.date) -> list[np.ndarray]:
    """Per-location series split at long gaps, keeping segments of at least a year."""
    out = []
    df = df[df["date"] < pd.Timestamp(cutoff)].dropna(subset=["value"])
    for _, g in df.groupby("location"):
        s = g.drop_duplicates("date").set_index("date")["value"].sort_index().astype(float)
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq=FREQ)).interpolate(
            limit=C.MAX_GAP_WEEKS, limit_area="inside")
        breaks = s.isna().to_numpy()
        seg_id = np.cumsum(breaks)
        for _, part in s[~breaks].groupby(seg_id[~breaks]):
            if len(part) >= MIN_SEGMENT_WEEKS:
                out.append(part.to_numpy(np.float32))
    return out


def training_corpus(cutoff: dt.date, include_ilinet: bool = True) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(training series, validation series). Validation = NHSN series; training excludes their last 8 weeks."""
    locations = data.load_locations()
    nhsn = _segments(data.fetch_nhsn_history(locations), cutoff)
    train = [s[:-8] for s in nhsn if len(s) > MIN_SEGMENT_WEEKS + 8]
    if include_ilinet:
        ili = _segments(data.fetch_ilinet_history(locations), cutoff)
        log.info("ILINet: %d segments", len(ili))
        train += ili
    log.info("Training corpus: %d series, %d weeks; validation: %d series",
             len(train), sum(map(len, train)), len(nhsn))
    return train, nhsn


def finetune_chronos2(cutoff: dt.date, out_dir: Path, steps: int = 1000, learning_rate: float = 1e-5,
                      mode: str = "lora", batch_size: int = 64, prediction_length: int = 8,
                      include_ilinet: bool = True) -> Path:
    from chronos import Chronos2Pipeline

    train, val = training_corpus(cutoff, include_ilinet)
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device())
    pipe.fit(
        inputs=train,
        validation_inputs=val,
        prediction_length=prediction_length,
        finetune_mode=mode,
        learning_rate=learning_rate,
        num_steps=steps,
        batch_size=batch_size,
        output_dir=str(out_dir),
    )
    ckpt = Path(out_dir) / "finetuned-ckpt"
    (Path(out_dir) / "finetune.json").write_text(json.dumps({
        "cutoff": str(cutoff), "steps": steps, "learning_rate": learning_rate, "mode": mode,
        "batch_size": batch_size, "prediction_length": prediction_length, "include_ilinet": include_ilinet,
        "n_train_series": len(train), "created": dt.datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    log.info("Fine-tuned checkpoint saved to %s", ckpt)
    return ckpt
