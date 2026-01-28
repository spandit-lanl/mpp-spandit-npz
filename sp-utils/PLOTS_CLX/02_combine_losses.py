#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re

# Supports:
#   loss_pretrain_L_01_lr-5_opt-adam_wd-3.csv
#   loss_direct_train_lsc_L_01_lr-5_opt-sgd_wd-3.csv
FNAME_RE = re.compile(
    r"^loss_(?P<phase>pretrain|finetune|direct_train|common_pretrain)"
    r"(?:_(?P<prefix>[A-Za-z0-9]+))?"   # optional prefix (e.g. lsc)
    r"_(?P<variant>L|B)"                # model size / variant
    r"_(?:nsteps_(?P<ns_old>\d+)|ns-(?P<ns_ns>\d+)|(?P<ns_plain>\d+))"
    r"(?:_(?P<rest>.*))?\.csv$"         # rest (lr/opt/wd etc.)
)

def parse_args():
    p = argparse.ArgumentParser(description="Combine per-run loss CSVs into one wide CSV.")
    p.add_argument(
        "--pattern",
        default="**/loss_*.csv",
        help="Glob for loss csvs. Default searches recursively: **/loss_*.csv",
    )
    p.add_argument("--output", default="combined_losses.csv")
    return p.parse_args()

def parse_filename(basename: str):
    """
    Supported filename patterns include (examples):
      - loss_pretrain_L_01_lr-3_opt-adam_wd-3.csv
      - loss_direct_train_L_01_lr-X.csv
      - loss_common_pretrain_L_ns-01_lr-X.csv   (treated as phase='pretrain')
      - loss_pretrain_b_MPP_nsteps_01.csv       (legacy)
    """
    m = FNAME_RE.match(basename)
    if not m:
        return None

    gd = m.groupdict()

    # Normalize phase tokens:
    # - "common_pretrain" is just a naming convention; treat it as "pretrain"
    phase = gd["phase"]
    if phase == "common_pretrain":
        phase = "pretrain"

    prefix = gd.get("prefix") or ""
    variant = gd["variant"]

    # ns/run id can come from multiple historical formats
    ns_str = gd.get("ns_old") or gd.get("ns_ns") or gd.get("ns_plain")
    ns = int(ns_str)

    rest = gd.get("rest") or ""

    # Tag base includes prefix if present
    if prefix:
        tag_base = f"{phase}_{prefix}_{variant}_{ns:02d}"
    else:
        tag_base = f"{phase}_{variant}_{ns:02d}"

    tag = tag_base + (f"_{rest}" if rest else "")

    phase_key = {"pretrain": 0, "finetune": 1, "direct_train": 2}[phase]
    sort_key = (phase_key, prefix, variant, ns, rest)
    return sort_key, tag, phase

def main():
    args = parse_args()

    files = sorted(glob.glob(args.pattern, recursive=True))
    if not files:
        raise SystemExit(f"No files match pattern {args.pattern!r}")

    all_epochs = set()
    runs = []
    skipped = []

    for path in files:
        base = os.path.basename(path)
        parsed = parse_filename(base)
        if not parsed:
            skipped.append(base)
            continue

        sort_key, tag, phase = parsed
        data = {}

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "Epoch" not in reader.fieldnames:
                skipped.append(base)
                continue

            for row in reader:
                epoch = int(row["Epoch"])
                train = (row.get("TrainLoss") or "").strip()
                valid = (row.get("ValidLoss") or "").strip()
                data[epoch] = (train, valid)
                all_epochs.add(epoch)

        runs.append({"sort_key": sort_key, "tag": tag, "phase": phase, "data": data, "path": path})

    if not runs:
        raise SystemExit("No valid loss_*.csv files found (filenames did not match expected pattern).")

    runs.sort(key=lambda r: r["sort_key"])
    epochs_sorted = sorted(all_epochs)

    header = ["Epoch"]
    for r in runs:
        header += [f"{r['tag']}_TrainLoss", f"{r['tag']}_ValidLoss"]

    with open(args.output, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)

        for epoch in epochs_sorted:
            row = [epoch]
            for r in runs:
                vals = r["data"].get(epoch)
                if vals is None:
                    row.extend(["", ""])
                else:
                    train_str, valid_str = vals
                    train_out = train_str
                    if valid_str == "":
                        valid_out = ""
                    else:
                        try:
                            valid_out = f"{float(valid_str):.4f}"
                        except ValueError:
                            valid_out = valid_str
                    row.extend([train_out, valid_out])
            writer.writerow(row)

    # Debug summary (safe for phases with underscores like direct_train)
    phases = {"pretrain": 0, "finetune": 0, "direct_train": 0}
    for r in runs:
        phases[r["phase"]] += 1

    print(f"Wrote {args.output} with {len(epochs_sorted)} epochs and {len(runs)} runs.")
    print(f"Runs by phase: {phases}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) that didn't match expected pattern or columns.")

if __name__ == "__main__":
    main()

