#!/bin/bash

# Usage: ./run_nsteps.sh <nsteps>
# Example: ./run_nsteps.sh 4   -> uses "04"
#          ./run_nsteps.sh 12  -> uses "12"


# ========================
# Validate 'n_steps' input
# ========================
VALID_N_STEPS=("1" "2" "4" "6" "8" "10" "12" "14" "16")

if [ $# -ne 1 ]; then
  echo "❌ Error: Missing required argument 'n_steps'."
  echo "✅ Usage: $0 <n_steps>     (Allowed values: 1, 2, 4, 8, 12, 16)"
  exit 1
fi

n_steps="$1"
# Convert to two digits (e.g. 1 -> 01, 9 -> 09, 12 -> 12)
n_steps=$(printf "%02d" $1)

if [[ ! " ${VALID_N_STEPS[@]} " =~ " ${n_steps} " ]]; then
  echo "❌ Error: Invalid value for 'n_steps': ${n_steps}"
  echo "✅ Allowed values: 1, 2, 4, 8, 12, 16"
  exit 1
fi

# ========================
# Training Config
# ========================

# Possible Datasets
DS_FP16_HALF='lsc240420_fp16_half'
DS_FP16_FULL='lsc240420_fp16_full'
DS_FP64_HALF='lsc240420_half'
DS_FP64_FULL='lsc240420'

# ========================
# Training Launch
# All trainings are run from scratch5
# ========================
python train_basic.py \
  --run_name lsc240420_nsteps_${n_steps} \
  --config basic_config \
  --yaml_config config/mpp_lsc_avit_ti_config_nsteps_${n_steps}.yaml \
  &>> output_lsc240420_nsteps_${n_steps}.txt

