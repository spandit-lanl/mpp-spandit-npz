#!/bin/bash
set -euo pipefail

./01_extract_losses.bash
python3 02_combine_losses.py
python3 03_verify_combines_losses.py

python3 04_plot_grid_losses.py --phase pretrain      --output pretrain_grid.png
python3 04_plot_grid_losses.py --phase finetune      --output finetune_grid.png
python3 04_plot_grid_losses.py --phase direct_train  --output direct_train_grid.png

