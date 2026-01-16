#!/usr/bin/env python3
import argparse
import os
import random
import re
import shutil

# Regex for filenames like: lsc240420_idXXXXX_pvi_idx00YYY.npz
FNAME_RE = re.compile(r"^lsc240420_id(\d{5})_pvi_idx(\d{5})\.npz$")


def parse_args():
    p = argparse.ArgumentParser(
        description="Randomly split experiments (XXXXX) and copy .npz files."
    )
    p.add_argument("--input", required=True, help="Input directory with .npz files")
    p.add_argument("--train_out", required=True, help="Output dir for train (e.g. 50%)")
    p.add_argument("--test_out", required=True, help="Output dir for test (e.g. 15%)")
    p.add_argument("--train_pct", type=float, required=True,
                   help="Percent of experiments for train (e.g. 50)")
    p.add_argument("--test_pct", type=float, required=True,
                   help="Percent of REMAINING experiments for test (e.g. 30)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (optional, for reproducibility)")
    return p.parse_args()


def collect_experiment_ids(input_dir):
    """Scan directory once and collect unique XXXXX experiment IDs."""
    exp_ids = set()
    with os.scandir(input_dir) as it:
        for entry in it:
            if not entry.is_file() or not entry.name.endswith(".npz"):
                continue
            m = FNAME_RE.match(entry.name)
            if m:
                exp_ids.add(m.group(1))  # XXXXX
    return sorted(exp_ids)


def choose_splits(exp_ids, train_pct, test_pct):
    """Randomly select train and test experiment IDs."""
    n_total = len(exp_ids)
    n_train = int(round(n_total * train_pct / 100.0))
    train_ids = set(random.sample(exp_ids, n_train))

    remaining = [e for e in exp_ids if e not in train_ids]
    n_test = int(round(len(remaining) * test_pct / 100.0))
    test_ids = set(random.sample(remaining, n_test))

    return train_ids, test_ids


def copy_files(input_dir, out_dir, selected_ids):
    """Copy all files whose XXXXX is in selected_ids."""
    os.makedirs(out_dir, exist_ok=True)
    with os.scandir(input_dir) as it:
        for entry in it:
            if not entry.is_file() or not entry.name.endswith(".npz"):
                continue
            m = FNAME_RE.match(entry.name)
            if not m:
                continue
            exp_id = m.group(1)
            if exp_id in selected_ids:
                src = os.path.join(input_dir, entry.name)
                dst = os.path.join(out_dir, entry.name)
                shutil.copy2(src, dst)


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    exp_ids = collect_experiment_ids(args.input)
    if not exp_ids:
        raise SystemExit("No matching .npz files found in input directory.")

    train_ids, test_ids = choose_splits(exp_ids, args.train_pct, args.test_pct)

    print(f"Total experiments: {len(exp_ids)}")
    print(f"Train experiments: {len(train_ids)}")
    print(f"Test experiments:  {len(test_ids)}")

    print("Copying train files...")
    copy_files(args.input, args.train_out, train_ids)

    print("Copying test files...")
    copy_files(args.input, args.test_out, test_ids)

    print("Done.")


if __name__ == "__main__":
    main()

