#!/usr/bin/env python3
"""
Plot train + validation losses on a 4x4 grid from combined_losses.csv.

Each subplot corresponds to one context window size (nsteps).
You can filter by phase (pretrain/finetune) and by nsteps parity (all/even/odd).

Behavior:
- All subplots share the same x/y scales (sharex=True, sharey=True).
- Two filters are used when deciding whether to plot a point:

  1) Global outlier filter:
     - Compute the median of all non-None losses (train + valid) across the
       selected runs.
     - Define a cutoff = median * 5 (fallback to 99th percentile if needed).
     - Any point with loss > cutoff is ignored.

  2) Per-run early-epoch filter:
     - For each run and for each series (train / valid):
       * Compute the average loss over epochs 1–5 (or however many of
         those epochs are present and non-None).
       * For epochs > 5, skip any point where loss > (average of epochs 1–5).

  Only points that pass BOTH filters are plotted, and only those values are used
  to set the y-axis max. This prevents late spikes from destroying the scale.
"""

import argparse
import csv
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Column naming pattern from combined_losses.csv:
#   pretrain_n01_TrainLoss
#   pretrain_n01_ValidLoss
#   finetune_n05_TrainLoss
COL_RE = re.compile(r"^(pretrain|finetune)_n(\d+)_([A-Za-z]+)$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="combined_losses.csv",
        help="Path to combined losses CSV (default: combined_losses.csv)",
    )
    p.add_argument(
        "--phase",
        required=True,
        choices=["pretrain", "finetune"],
        help="Which phase to plot (pretrain or finetune)",
    )
    p.add_argument(
        "--which",
        choices=["all", "even", "odd"],
        default="all",
        help="Which nsteps to include: all, even, or odd (default: all)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output PNG filename (default: auto from phase + which)",
    )

    # Layout knobs (useful when you want the title/legend closer to the top row)
    p.add_argument(
        "--suptitle-y",
        type=float,
        default=0.95,
        help=(
            "Figure y-position for the main title (0–1, default: 0.95). "
            "Smaller values move the title closer to the subplots."
        ),
    )
    p.add_argument(
        "--legend-y",
        type=float,
        default=0.95,
        help=(
            "Figure y-position for the global legend anchor (0–1, default: 0.95). "
            "Smaller values move the legend closer to the subplots."
        ),
    )
    p.add_argument(
        "--tight-top",
        type=float,
        default=0.94,
        help=(
            "Top of the tight_layout rect (0–1, default: 0.94). "
            "Larger values move the subplot grid upward."
        ),
    )
    return p.parse_args()


def load_combined_for_phase(path, phase):
    """
    Load combined_losses.csv and extract runs for a given phase.

    Returns:
      epochs: list[int]
      runs_by_nstep: dict[int] -> {"train": [float|None], "valid": [float|None]}
    """
    if not os.path.exists(path):
        raise SystemExit(f"Input file {path!r} not found.")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    if not header or "Epoch" not in header:
        raise ValueError("Expected an 'Epoch' column in the CSV header.")

    # Identify columns belonging to the requested phase, grouped by nstep.
    meta_by_nstep = {}  # nstep -> {"train_col": str, "valid_col": str}
    for col in header:
        if col == "Epoch":
            continue
        m = COL_RE.match(col)
        if not m:
            continue
        col_phase, nstep_str, kind = m.groups()
        if col_phase != phase:
            continue
        nstep = int(nstep_str)
        meta = meta_by_nstep.setdefault(nstep, {"train_col": None, "valid_col": None})
        kind_lower = kind.lower()
        if kind_lower.startswith("train"):
            meta["train_col"] = col
        elif kind_lower.startswith("valid"):
            meta["valid_col"] = col

    epochs = [int(r["Epoch"]) for r in rows]

    runs_by_nstep = {}
    for nstep, meta in meta_by_nstep.items():
        tcol = meta["train_col"]
        vcol = meta["valid_col"]
        train_vals = []
        valid_vals = []
        for r in rows:
            if tcol is not None:
                s = (r.get(tcol) or "").strip()
                train_vals.append(float(s) if s else None)
            else:
                train_vals.append(None)
            if vcol is not None:
                s = (r.get(vcol) or "").strip()
                valid_vals.append(float(s) if s else None)
            else:
                valid_vals.append(None)

        runs_by_nstep[nstep] = {"train": train_vals, "valid": valid_vals}

    return epochs, runs_by_nstep


def main():
    args = parse_args()

    epochs, runs_by_nstep = load_combined_for_phase(args.input, args.phase)
    if not runs_by_nstep:
        raise SystemExit(f"No runs found for phase={args.phase!r} in {args.input}")

    # Filter nsteps by parity if requested
    nsteps = sorted(runs_by_nstep.keys())
    if args.which == "even":
        nsteps = [n for n in nsteps if n % 2 == 0]
    elif args.which == "odd":
        nsteps = [n for n in nsteps if n % 2 == 1]

    if not nsteps:
        raise SystemExit(f"No nsteps remaining after applying filter which={args.which!r}")

    # Limit to at most 16 subplots
    if len(nsteps) > 16:
        print(f"Warning: {len(nsteps)} nsteps found, plotting only the first 16.")
        nsteps = nsteps[:16]

    if args.output is None:
        args.output = f"{args.phase}_{args.which}_grid.png"

    # --- Global outlier utoff based on median across all selected runs ---
    all_vals = []
    for n in nsteps:
        data = runs_by_nstep[n]
        for v in data["train"]:
            if v is not None:
                all_vals.append(v)
        for v in data["valid"]:
            if v is not None:
                all_vals.append(v)

    if not all_vals:
        raise SystemExit("No loss values found to plot.")

    all_vals_arr = np.array(all_vals, dtype=float)
    median_val = float(np.median(all_vals_arr))
    factor = 10
    cutoff = median_val * factor
    if cutoff <= 0:
        cutoff = float(np.quantile(all_vals_arr, 0.99))

    print(f"Global median loss = {median_val:.4g}, global cutoff = {cutoff:.4g}")

    # --- Build filtered series per run, and collect all plotted y-values to set y_max ---
    series_by_nstep = {}
    all_plotted_vals = []

    for n in nsteps:
        data = runs_by_nstep[n]
        train_vals = data["train"]
        valid_vals = data["valid"]

        # Per-run averages of epochs 1–5 (index 0..4), ignoring None.
        train_early = [t for t in train_vals[:5] if t is not None]
        valid_early = [v for v in valid_vals[:5] if v is not None]
        train_avg_early = sum(train_early) / len(train_early) if train_early else None
        valid_avg_early = sum(valid_early) / len(valid_early) if valid_early else None

        train_x, train_y = [], []
        valid_x, valid_y = [], []

        for idx, (e, t, v) in enumerate(zip(epochs, train_vals, valid_vals)):
            # Train series
            if t is not None:
                # Always apply global cutoff
                if t <= cutoff:
                    # For first 5 epochs, always keep (after cutoff)
                    if idx < 5 or train_avg_early is None or t <= train_avg_early:
                        train_x.append(e)
                        train_y.append(t)
                        all_plotted_vals.append(t)

            # Valid series
            if v is not None:
                if v <= cutoff:
                    if idx < 5 or valid_avg_early is None or v <= valid_avg_early:
                        valid_x.append(e)
                        valid_y.append(v)
                        all_plotted_vals.append(v)

        series_by_nstep[n] = {
            "train_x": train_x,
            "train_y": train_y,
            "valid_x": valid_x,
            "valid_y": valid_y,
        }

    if not all_plotted_vals:
        raise SystemExit("After filtering, no points remain to plot.")

    y_max = max(all_plotted_vals) * 1.05

    # --- Plotting ---
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax in axes:
        ax.set_visible(False)

    for idx, n in enumerate(nsteps):
        ax = axes[idx]
        ax.set_visible(True)

        s = series_by_nstep[n]

        if s["train_x"]:
            ax.plot(s["train_x"], s["train_y"], label="Train")
        if s["valid_x"]:
            ax.plot(s["valid_x"], s["valid_y"], label="Valid")

        ax.set_title(f"n = {n}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=y_max)

        # Show integer epoch numbers on every subplot
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="x", labelbottom=True)

    # Shared labels
    fig.text(0.5, 0.04, "Epoch", ha="center")
    fig.text(0.04, 0.5, "Loss", va="center", rotation="vertical")

    # Main title (moved a bit lower by default via --suptitle-y)
    fig.suptitle(
        f"{args.phase.capitalize()} — {args.which} nsteps (Train & Validation Loss)",
        fontsize=16,
        y=args.suptitle_y,
    )

    # Global legend
    handles, labels = None, None
    for ax in axes:
        if not ax.get_visible():
            continue
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        # Global legend (also moved down by default via --legend-y)
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.985, args.legend_y),
            borderaxespad=0.0,
        )

    # Raise the subplot grid a bit (default: 0.94) so the top row sits closer to the
    # title/legend. You can tweak this via --tight-top.
    plt.tight_layout(rect=[0.05, 0.05, 0.98, args.tight_top])
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}, y_max ≈ {y_max:.4g}")


if __name__ == "__main__":
    main()

