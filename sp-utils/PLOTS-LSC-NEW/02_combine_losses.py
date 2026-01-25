#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re

# loss_pretrain_L_01_lr-5_opt-adam_wd-3.csv
FNAME_RE = re.compile(
    r"^loss_(pretrain|finetune|direct_train)_([A-Za-z0-9]+)_(\d+)(?:_(.*))?\.csv$"
)

def parse_args():
    p = argparse.ArgumentParser(description="Combine per-run loss CSVs into a single wide CSV.")
    p.add_argument("--pattern", default="loss_*.csv")
    p.add_argument("--output", default="combined_losses.csv")
    return p.parse_args()

def parse_filename(name: str):
    m = FNAME_RE.match(name)
    if not m:
        return None
    phase, variant, runid_str, extra = m.groups()
    runid = int(runid_str)
    extra = extra or ""
    # tag becomes: pretrain_L_01_lr-5_opt-adam_wd-3
    tag = f"{phase}_{variant}_{runid:02d}" + (f"_{extra}" if extra else "")
    # sort by phase order, then variant, then runid, then extra
    phase_key = {"pretrain": 0, "finetune": 1, "direct_train": 2}[phase]
    sort_key = (phase_key, variant, runid, extra)
    return sort_key, tag

def main():
    args = parse_args()
    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files match pattern {args.pattern!r}")

    all_epochs = set()
    runs = []

    for path in files:
        name = os.path.basename(path)
        parsed = parse_filename(name)
        if not parsed:
            continue

        sort_key, tag = parsed
        data = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = int(row["Epoch"])
                train = row["TrainLoss"].strip()
                valid = row["ValidLoss"].strip()
                data[epoch] = (train, valid)
                all_epochs.add(epoch)

        runs.append({"sort_key": sort_key, "tag": tag, "data": data})

    if not runs:
        raise SystemExit("No valid loss_*.csv files found.")

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
                    # TrainLoss: keep as-is
                    train_out = train_str
                    # ValidLoss: round to 4 decimals if possible
                    if valid_str == "":
                        valid_out = ""
                    else:
                        try:
                            valid_out = f"{float(valid_str):.4f}"
                        except ValueError:
                            valid_out = valid_str
                    row.extend([train_out, valid_out])
            writer.writerow(row)

    print(f"Wrote {args.output} with {len(epochs_sorted)} epochs and {len(runs)} runs.")

if __name__ == "__main__":
    main()

