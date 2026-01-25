#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


COL_RE = re.compile(r"^(pretrain|finetune|direct_train)_(.+)_(TrainLoss|ValidLoss)$")
TAG_RE = re.compile(
    r"^(?P<variant>[A-Za-z0-9]+)_(?P<ns>\d+)"
    r"(?:_lr-(?P<lr>[0-9A-Za-z]+))?"
    r"(?:_opt-(?P<opt>adam|adan|sgd))?"
    r"(?:_wd-(?P<wd>[0-9]+))?$"
)

@dataclass(frozen=True)
class RunInfo:
    phase: str
    tag: str
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

def lr_sort_key(lr: str) -> Tuple[int, str]:
    try:
        return (0, f"{int(lr):04d}")
    except Exception:
        return (1, lr)

def eight_ticks(max_epoch: int) -> List[int]:
    """
    Return 8 tick positions, evenly spaced, excluding 0, always including max_epoch.
    Examples:
      max=1000 -> 125,250,...,1000
      max=2000 -> 250,500,...,2000
    """
    if max_epoch <= 0:
        return [1,2,3,4,5,6,7,8]
    step = max(1, max_epoch // 8)
    ticks = [step * i for i in range(1, 9)]
    # Ensure last tick is exactly max_epoch
    ticks[-1] = max_epoch
    # Ensure strictly increasing (can break if max_epoch < 8)
    ticks = sorted(set(ticks))
    # If we ended up with <8 ticks (small max_epoch), pad evenly
    while len(ticks) < 8:
        # insert midpoints roughly
        candidate = max_epoch * (len(ticks) + 1) // 9
        if candidate not in ticks and candidate > 0:
            ticks.append(candidate)
        else:
            # fallback incremental
            nxt = ticks[-1] + 1
            ticks.append(nxt)
        ticks = sorted(set(ticks))
    # If >8 ticks (due to set), downsample evenly
    if len(ticks) > 8:
        idxs = [round(i * (len(ticks) - 1) / 7) for i in range(8)]
        ticks = [ticks[i] for i in idxs]
        ticks[-1] = max_epoch
    return ticks

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot loss curves in a grid: one subplot per (lr,opt,wd) for a given phase/ns/variant."
    )
    p.add_argument("--input", default="combined_losses.csv")
    p.add_argument("--phase", required=True, choices=["pretrain", "finetune", "direct_train"])
    p.add_argument("--variant", default=None)
    p.add_argument("--ns", default=None)
    p.add_argument("--output", default=None)

    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--max_plots", type=int, default=32)
    p.add_argument("--sort", choices=["lr_opt_wd", "opt_lr_wd"], default="lr_opt_wd")
    return p.parse_args()

def tag_to_runinfo(phase: str, tag: str) -> Optional[RunInfo]:
    m = TAG_RE.match(tag)
    if not m:
        return None
    gd = m.groupdict()
    return RunInfo(
        phase=phase,
        tag=tag,
        variant=gd.get("variant") or "",
        ns=gd.get("ns") or "",
        lr=gd.get("lr") or "?",
        opt=gd.get("opt") or "?",
        wd=gd.get("wd") or "?",
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

    epochs = [int(r["Epoch"]) for r in rows]
    max_epoch = max(epochs) if epochs else 0
    xticks = eight_ticks(max_epoch)

    # Discover all run tags for this phase
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

    run_infos = sorted(set(run_infos), key=lambda x: x.tag)

    # Filters
    if args.variant is not None:
        run_infos = [ri for ri in run_infos if ri.variant == args.variant]
    if args.ns is not None:
        ns_norm = args.ns.zfill(2) if args.ns.isdigit() else args.ns
        run_infos = [ri for ri in run_infos if ri.ns == ns_norm]

    if not run_infos:
        raise SystemExit(
            f"No runs found after filtering for phase={args.phase!r}, "
            f"variant={args.variant!r}, ns={args.ns!r}."
        )

    # One panel per (lr,opt,wd)
    groups: Dict[Tuple[str, str, str], RunInfo] = {}
    for ri in run_infos:
        groups.setdefault((ri.lr, ri.opt, ri.wd), ri)

    keys = list(groups.keys())
    if args.sort == "lr_opt_wd":
        keys.sort(key=lambda k: (lr_sort_key(k[0]), k[1], int(k[2]) if k[2].isdigit() else 9999))
    else:
        keys.sort(key=lambda k: (k[1], lr_sort_key(k[0]), int(k[2]) if k[2].isdigit() else 9999))

    keys = keys[: args.max_plots]

    if args.output is None:
        parts = [args.phase]
        if args.variant: parts.append(args.variant)
        if args.ns: parts.append(f"ns{args.ns}")
        args.output = "_".join(parts) + "_grid.png"

    # Precompute global y-range across all selected panels (both Train+Valid)
    y_min = float("inf")
    y_max = float("-inf")

    def update_y_range(colname: str):
        nonlocal y_min, y_max
        for r in rows:
            s = (r.get(colname) or "").strip()
            if not s:
                continue
            v = safe_float(s)
            if v is None:
                continue
            if v < y_min: y_min = v
            if v > y_max: y_max = v

    for (lr, opt, wd) in keys:
        ri = groups[(lr, opt, wd)]
        tcol = f"{ri.phase}_{ri.tag}_TrainLoss"
        vcol = f"{ri.phase}_{ri.tag}_ValidLoss"
        if tcol in header: update_y_range(tcol)
        if vcol in header: update_y_range(vcol)

    if y_min == float("inf") or y_max == float("-inf"):
        raise SystemExit("Could not compute global y-range (no numeric values found).")

    # Add a small padding
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    y_lo = max(0.0, y_min - pad)
    y_hi = y_max + pad

    ncols = max(1, args.ncols)
    nrows = math.ceil(len(keys) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.8 * ncols, 3.6 * nrows),
        sharex=True,
        sharey=True  # force shared Y scale
    )
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = list(axes.ravel())

    for ax in axes:
        ax.set_visible(False)

    for i, (lr, opt, wd) in enumerate(keys):
        ri = groups[(lr, opt, wd)]
        ax = axes[i]
        ax.set_visible(True)

        tcol = f"{ri.phase}_{ri.tag}_TrainLoss"
        vcol = f"{ri.phase}_{ri.tag}_ValidLoss"

        train_x, train_y = [], []
        valid_x, valid_y = [], []

        for e, r in zip(epochs, rows):
            ts = (r.get(tcol) or "").strip()
            vs = (r.get(vcol) or "").strip()
            tv = safe_float(ts) if ts else None
            vv = safe_float(vs) if vs else None
            if tv is not None:
                train_x.append(e); train_y.append(tv)
            if vv is not None:
                valid_x.append(e); valid_y.append(vv)

        ax.plot(train_x, train_y, label="Train")
        ax.plot(valid_x, valid_y, label="Valid")

        #ax.set_title(f"{ri.phase} {ri.variant} ns={ri.ns} lr={lr} opt={opt} wd={wd}", fontsize=11)
        ax.set_title( rf"{ri.phase} {ri.variant} ns={ri.ns} $\mathbf{{lr={lr}\ opt={opt}\ wd={wd}}}$", fontsize=11)
        ax.grid(True, alpha=0.3)

        # Global axis settings
        ax.set_xlim(0, max_epoch)
        ax.set_xticks(xticks)
        ax.set_ylim(y_lo, y_hi)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"{args.phase} — Train & Valid Loss (grid by lr/opt/wd)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()

