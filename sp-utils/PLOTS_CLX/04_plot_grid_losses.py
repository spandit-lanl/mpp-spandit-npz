#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# MATCHES YOUR HEADER:
#   direct_train_L_01_lr-X_TrainLoss
#   pretrain_L_08_lr-X_ValidLoss
COL_RE = re.compile(
    r"^(?P<phase>pretrain|finetune|direct_train)_"
    r"(?P<ms>B|L)_"
    r"(?P<ns>\d{2})_"
    r"lr-(?P<lr>[0-9X]+)_"
    r"(?P<kind>TrainLoss|ValidLoss)$"
)

def safe_float(s: str):
    try:
        return float(s)
    except Exception:
        return None

def eight_ticks(max_epoch: int):
    """Exactly 8 ticks excluding 0, ending at max_epoch."""
    if max_epoch <= 0:
        return [1, 2, 3, 4, 5, 6, 7, 8]
    step = max(1, max_epoch // 8)
    ticks = [step * i for i in range(1, 9)]
    ticks[-1] = max_epoch
    ticks = sorted(set(ticks))
    while len(ticks) < 8:
        cand = int(round(max_epoch * (len(ticks) + 1) / 9))
        if cand > 0:
            ticks.append(cand)
        ticks = sorted(set(ticks))
        if len(ticks) < 8 and ticks[-1] < max_epoch:
            ticks.append(min(max_epoch, ticks[-1] + 1))
            ticks = sorted(set(ticks))
    if len(ticks) > 8:
        idxs = [round(i * (len(ticks) - 1) / 7) for i in range(8)]
        ticks = [ticks[i] for i in idxs]
        ticks[-1] = max_epoch
    return ticks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="combined_losses.csv")
    ap.add_argument("--phase", required=True, choices=["pretrain", "finetune", "direct_train"])
    ap.add_argument("--variant", default="all", choices=["all", "B", "L"], help="Model size filter")
    ap.add_argument("--which", default="all", choices=["all", "TrainLoss", "ValidLoss"])
    ap.add_argument("--ncols", type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    if "Epoch" not in header:
        raise SystemExit("CSV must contain an 'Epoch' column.")

    epochs = np.array([int(r["Epoch"]) for r in rows], dtype=int)
    max_epoch = int(np.max(epochs)) if epochs.size else 0
    xticks = eight_ticks(max_epoch)

    # key = (ms, ns, lr) -> {"TrainLoss": col, "ValidLoss": col}
    runs = defaultdict(dict)
    for c in header:
        m = COL_RE.match(c)
        if not m:
            continue
        gd = m.groupdict()
        if gd["phase"] != args.phase:
            continue
        if args.variant != "all" and gd["ms"] != args.variant:
            continue
        key = (gd["ms"], gd["ns"], gd["lr"])
        runs[key][gd["kind"]] = c

    if not runs:
        phase_cols = [c for c in header if c.startswith(args.phase + "_")]
        raise SystemExit(
            f"No matching columns found for phase={args.phase!r}, variant={args.variant!r}.\n"
            f"Columns starting with '{args.phase}_': {len(phase_cols)}"
        )

    keys = sorted(runs.keys(), key=lambda k: (k[0], k[1], k[2]))

    # Global y-range from what we will plot
    y_min, y_max = float("inf"), float("-inf")

    def update_range(col):
        nonlocal y_min, y_max
        for r in rows:
            v = safe_float((r.get(col) or "").strip())
            if v is None or not math.isfinite(v):
                continue
            y_min = min(y_min, v)
            y_max = max(y_max, v)

    for k in keys:
        cols = runs[k]
        if args.which in ("all", "TrainLoss") and "TrainLoss" in cols:
            update_range(cols["TrainLoss"])
        if args.which in ("all", "ValidLoss") and "ValidLoss" in cols:
            update_range(cols["ValidLoss"])

    if not (math.isfinite(y_min) and math.isfinite(y_max)):
        raise SystemExit("Could not compute Y range (no numeric loss values).")

    pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    y_lo = max(0.0, y_min - pad)
    y_hi = y_max + pad

    n = len(keys)
    ncols = max(1, min(args.ncols, n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4 * ncols, 3 * nrows),
        squeeze=False, sharex=True, sharey=True
    )

    for idx, k in enumerate(keys):
        ms, ns, lr = k
        ax = axes[idx // ncols][idx % ncols]
        cols = runs[k]

        if args.which in ("all", "TrainLoss") and "TrainLoss" in cols:
            y = np.array([safe_float((r.get(cols["TrainLoss"]) or "").strip()) for r in rows], dtype=float)
            ax.plot(epochs, y, label="Train")

        if args.which in ("all", "ValidLoss") and "ValidLoss" in cols:
            y = np.array([safe_float((r.get(cols["ValidLoss"]) or "").strip()) for r in rows], dtype=float)
            ax.plot(epochs, y, label="Valid")

        ax.set_title(f"{args.phase} {ms} ns={ns} lr-{lr}")
        ax.set_xlim(0, max_epoch)
        ax.set_xticks(xticks)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # CLX output naming logic (as requested earlier)
    csv_base = os.path.basename(args.input).lower()
    if "direct" in csv_base:
        fig.suptitle("Direct Training with Cylex Data", fontsize=24, fontweight="bold", y=0.995)
        out = "CLX_direct.png"
    else:
        fig.suptitle("Pretrain on PDEBench Data", fontsize=24, fontweight="bold", y=0.995)
        out = "CLX_pretrain.png"

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()

