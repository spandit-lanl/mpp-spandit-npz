#!/bin/bash

./01_extract_losses.bash
python3 02_combine_losses.py
python3 03_verify_combines_losses.py

python3 04_plot_grid_losses.py --phase pretrain
yes y | cp -f combined_losses.csv combined_losses_direct_train.csv
python3 04_plot_grid_losses.py --input combined_losses_direct_train.csv --phase direct_train

#python3 04_plot_grid_losses.py --phase finetune --which odd --output finetune_odd_grid.png
#python3 04_plot_grid_losses.py --phase pretrain --which odd --output pretrain_odd_grid.png

#python3 04_plot_grid_losses.py --phase pretrain --which even --output pretrain_even_grid.png
#python3 04_plot_grid_losses.py --phase finetune --which even --output finetune_even_grid.png



