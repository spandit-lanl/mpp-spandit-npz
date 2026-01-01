#!/bin/bash

#All trainings are run from scratch5

############################################################
# Check that exactly two arguments are provided
############################################################
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <nsteps (1-16)> <pretrain|finetune>"
  exit 1
fi

NSTEPS_INPUT="$1"
PREFIX_MODE="$2"

##################################################
# Validate range
##################################################
if [ "$NSTEPS_INPUT" -lt 1 ] || [ "$NSTEPS_INPUT" -gt 16 ]; then
  echo "Error: nsteps must be between 1 and 16"
  exit 1
fi


############################################################
# Validate prefix mode
############################################################
if [ "$PREFIX_MODE" != "pretrain" ] && [ "$PREFIX_MODE" != "finetune" ]; then
  echo "Error: prefix must be 'pretrain' or 'finetune'"
  exit 1
fi

##################################################
# Format as two digits (01–16)
##################################################
NSTEPS=$(printf "%02d" "$NSTEPS_INPUT")

############################################################
# Set prefix based on mode
############################################################

############################################################
############################################################
# Define Training and Fine TUning data
############################################################
############################################################
TRAIN_DATA='MPP'
FINETUNE_DATA='LSC'



if [ "$PREFIX_MODE" = "pretrain" ]; then
  RUN_PREFIX="train_${TRAIN_DATA}"
else
  RUN_PREFIX="train_${TRAIN_DATA}_finetune_${FINETUNE_DATA}"
fi

OUT_DIR="./OUT_${TRAIN_DATA}/${RUN_PREFIX}"

if [ ! -d ${OUT_DIR} ]; then
  mkdir -p ${OUT_DIR}
fi

############################################################
# Run Training
############################################################
echo "python train_basic.py                                                           \
  --run_name ${RUN_PREFIX}_nsteps_${NSTEPS}                                     \
  --config basic_config                                                         \
  --yaml_config config/config_${TRAIN_DATA}/mpp_avit_ti_config_nsteps_${NSTEPS}.yaml \
  &>> ${OUT_DIR}/out_${RUN_PREFIX}_nsteps_${NSTEPS}.txt
"
