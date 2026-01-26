#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

for f in ../out_*.log; do
  base=${f##*/}          # filename only
  base=${base%.log}

  # Strip leading "out_"
  stem="${base#out_}"

  # Keep only the 3 phases (allow extra tokens after phase, e.g. direct_train_lsc_...)
  if [[ ! "$stem" =~ ^(pretrain|finetune|direct_train)_ ]]; then
    continue
  fi

  outfile="loss_${stem}.csv"
  echo "writing $outfile"

  {
    echo "Epoch,TrainLoss,ValidLoss"
    awk '
      BEGIN { OFS=","; seq=0 }
      /Valid loss/ {
        seq++
        epoch = seq
        if (match($0, /Epoch:[[:space:]]*([0-9]+)/, m)) epoch = m[1]

        train = ""
        if (match($0, /Train loss:[[:space:]]*tensor\(\[([0-9.eE+-]+)/, t)) train = t[1]

        valid = ""
        if (match($0, /Valid loss:[[:space:]]*([0-9.eE+-]+)/, v)) valid = v[1]

        if (train != "" && valid != "")
          print epoch, train, valid
      }
    ' "$f"
  } > "$outfile"
done

rm -f *old.csv

