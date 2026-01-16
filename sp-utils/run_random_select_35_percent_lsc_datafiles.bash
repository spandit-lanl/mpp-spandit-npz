#!/bin/bash

python random_select_finetune_lsc_data.py                  \
    --input /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_half \
    --train_pct 35 \
    --test_pct 10 \
    --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune_on_lsc/lsc240420_fp16_half_train_35_percent_lsc \
    --test_out  /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune_on_lsc/lsc240420_fp16_half_test_10_percent_lsc

python random_select_finetune_lsc_data.py                  \
    --input /lustre/scratch5/exempt/artimis/data/lsc240420_fp16_full \
    --train_pct 35 \
    --test_pct 10 \
    --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune_on_lsc/lsc240420_fp16_full_train_35_percent_lsc \
    --test_out  /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune_on_lsc/lsc240420_fp16_full_test_10_percent_lsc

python random_select_finetune_lsc_data.py                  \
    --input /lustre/scratch5/exempt/artimis/data/lsc240420 \
    --train_pct 35 \
    --test_pct 10 \
    --train_out /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune_on_lsc/lsc240420_train_35_percent_lsc \
    --test_out  /lustre/scratch5/exempt/artimis/mpmm/spandit/data/mpp_finetune_on_lsc/lsc240420_test_10_percent_lsc

