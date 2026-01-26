#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Matches columns like:
#   pretrain_L_01_lr-5_opt-adam_wd-3_TrainLoss
#   direct_train_lsc_L_01_lr-5_opt-sgd_wd-3_ValidLoss
#
# IMPORTANT:
# - phase is ONLY the first token: pretrain/finetune/direct_train
# - tag is EVERYTHING between phase_ and _TrainLoss/_ValidLoss, including optional "lsc_"
COL_RE = re.compile(r"^(pretrain|finetune|direct_train)_(.+)_(TrainLoss|ValidLoss)$")


# Tag matches either:
#   L_01_lr-5_opt-adam_wd-3
#   lsc_L_01_lr-5_opt-adam_wd-3
TAG_RE = re.compile(
    r"^(?:(?P<prefix>[A-Za-z0-9]+)_)?"
    r"(?P<variant>[A-Za-z0-9]+)_(?P<ns>\d+)"
    r"_lr-(?P<lr>[0-9A-Za-z]+)"
    r"_opt-(?P<opt>adam|adan|sgd)"
    r"_wd-(?P<wd>[0-9]+)$"
)

@dataclass(frozen=True)
class RunInfo:
    phase: str
    tag: str
    prefix: str
    variant: str
    ns: str
    lr: str
    opt: str
    wd: str

def safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

def eight_ticks(max_epoch: int) -> List[int]:
    """
    Exactly 8 ticks excluding 0, evenly spaced, ending at max_epoch.
    Examples:
      max=1000 -> 125,250,...,1000
      max=2000 -> 250,500,...,2000
    """
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

def parse_args():
    p = argparse.ArgumentParser(description="Plot loss curves in a grid by (lr,opt,wd).")
    p.add_argument("--input", default="combined_losses.csv")
    p.add_argument("--phase", required=True, choices=["pretrain", "finetune", "direct_train"])
    p.add_argument("--variant", default=None, help="Filter variant (e.g. L). If omitted, includes all.")
    p.add_argument("--ns", default=None, help="Filter ns (e.g. 01). If omitted, includes all.")
    p.add_argument("--prefix", default=None, help="Filter prefix (e.g. lsc). If omitted, includes all.")
    p.add_argument("--output", default=None)
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--max_plots", type=int, default=32)
    return p.parse_args()

def tag_to_runinfo(phase: str, tag: str) -> Optional[RunInfo]:
    m = TAG_RE.match(tag)
    if not m:
        return None
    gd = m.groupdict()
    return RunInfo(
        phase=phase,
        tag=tag,
        prefix=gd.get("prefix") or "",
        variant=gd["variant"],
        ns=gd["ns"],
        lr=gd["lr"],
        opt=gd["opt"],
        wd=gd["wd"],
    )

def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise SystemExit(f"Input file {args.input!r} not found.")

    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    if "Epoch" not in header:
        raise SystemExit("Expected 'Epoch' column in combined CSV.")

    epochs = np.array([int(r["Epoch"]) for r in rows], dtype=int)
    max_epoch = int(np.max(epochs)) if epochs.size else 0
    xticks = eight_ticks(max_epoch)

    # Discover run tags for requested phase
    run_infos: List[RunInfo] = []
    for col in header:
        m = COL_RE.match(col)
        if not m:
            continue
        phase, tag, _kind = m.groups()
        if phase != args.phase:
            continue
        info = tag_to_runinfo(phase, tag)
        if info is not None:
            run_infos.append(info)

    # Deduplicate (Train/Valid)
    run_infos = sorted(set(run_infos), key=lambda x: x.tag)

    # Apply filters
    if args.variant is not None:
        run_infos = [ri for ri in run_infos if ri.variant == args.variant]
    if args.ns is not None:
        ns_norm = args.ns.zfill(2) if args.ns.isdigit() else args.ns
        run_infos = [ri for ri in run_infos if ri.ns == ns_norm]
    if args.prefix is not None:
        run_infos = [ri for ri in run_infos if ri.prefix == args.prefix]

    if not run_infos:
        # Helpful debug
        phase_cols = [c for c in header if c.startswith(args.phase + "_")]
        raise SystemExit(
            f"No runs found for phase={args.phase!r} after filters "
            f"(variant={args.variant!r}, ns={args.ns!r}, prefix={args.prefix!r}).\n"
            f"Columns starting with '{args.phase}_': {len(phase_cols)}\n"
            f"Example: {phase_cols[0] if phase_cols else None}"
        )

    # One subplot per (lr,opt,wd)
    groups: Dict[Tuple[str, str, str], RunInfo] = {}
    for ri in run_infos:
        groups.setdefault((ri.lr, ri.opt, ri.wd), ri)

    keys = sorted(groups.keys(), key=lambda k: (k[0], k[1], int(k[2])))
    keys = keys[: args.max_plots]

    if args.output is None:
        parts = [args.phase]
        if args.prefix: parts.append(args.prefix)
        if args.variant: parts.append(args.variant)
        if args.ns: parts.append(f"ns{args.ns}")
        args.output = "_".join(parts) + "_grid.png"

    # Global Y range across all plotted series
    y_min = float("inf")
    y_max = float("-inf")

    def update_range(colname: str):
        nonlocal y_min, y_max
        for r in rows:
            s = (r.get(colname) or "").strip()
            if not s:
                continue
            v = safe_float(s)
            if v is None:
                continue
            y_min = min(y_min, v)
            y_max = max(y_max, v)

    for (lr, opt, wd) in keys:
        ri = groups[(lr, opt, wd)]
        tcol = f"{ri.phase}_{ri.tag}_TrainLoss"
        vcol = f"{ri.phase}_{ri.tag}_ValidLoss"
        if tcol in header: update_range(tcol)
        if vcol in header: update_range(vcol)

    if not (math.isfinite(y_min) and math.isfinite(y_max)):
        raise SystemExit("Could not compute global Y range (no numeric values).")

    pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    y_lo = max(0.0, y_min - pad)
    y_hi = y_max + pad

    # Grid layout
    ncols = max(1, args.ncols)
    nrows = math.ceil(len(keys) / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 3.6 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()
    for ax in axes:
        ax.set_visible(False)

    # Big bold suptitle from last directory name
    dir_tag = os.path.basename(os.getcwd()).replace("-", "_")
    fig.suptitle(dir_tag, fontsize=24, fontweight="bold", y=0.995)

    # Plot each panel
    for i, (lr, opt, wd) in enumerate(keys):
        ri = groups[(lr, opt, wd)]
        ax = axes[i]
        ax.set_visible(True)

        tcol = f"{ri.phase}_{ri.tag}_TrainLoss"
        vcol = f"{ri.phase}_{ri.tag}_ValidLoss"

        train_y = np.array(
            [safe_float((r.get(tcol) or "").strip()) if (r.get(tcol) or "").strip() else np.nan for r in rows],
            dtype=float,
        )
        valid_y = np.array(
            [safe_float((r.get(vcol) or "").strip()) if (r.get(vcol) or "").strip() else np.nan for r in rows],
            dtype=float,
        )

        ax.plot(epochs, train_y, label="Train")
        ax.plot(epochs, valid_y, label="Valid")

        # ns NOT bold; lr/opt/wd bold
        ax.set_title(
            rf"{ri.phase} {ri.variant} ns={ri.ns} $\mathbf{{lr={lr}\ opt={opt}\ wd={wd}}}$",
            fontsize=11,
        )

        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_epoch)
        ax.set_xticks(xticks)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="best", fontsize=9)

    # Hide unused panels
    for j in range(len(keys), nrows * ncols):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output} (panels={len(keys)}, grid={nrows}x{ncols})")

if __name__ == "__main__":
    main()

