
#!/bin/bash

#All trainings are run from scratch5

##################################################
# Check that exactly one argument is provided
##################################################
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <nsteps (1-16)>"
  exit 1
fi

NSTEPS_INPUT="$1"

##################################################
# Validate range
##################################################
if [ "$NSTEPS_INPUT" -lt 1 ] || [ "$NSTEPS_INPUT" -gt 16 ]; then
  echo "Error: nsteps must be between 1 and 16"
  exit 1
fi

##################################################
# Format as two digits (01–16)
##################################################
NSTEPS=$(printf "%02d" "$NSTEPS_INPUT")

OUT_DIR='./OUT_TRAIN_LSC'

if [ ! -d ${OUT_DIR} ]; then
  mkdir ${OUT_DIR}
fi

RUN_PREFIX='train_LSC'
##################:################################
# Run Training
##################################################
python train_basic.py --run_name ${RUN_PREFIX}_nsteps_${NSTEPS} --config basic_config \
  --yaml_config config/config_${RUN_PREFIX}/mpp_avit_ti_config_nsteps_${NSTEPS}.yaml \
  &>> ${OUT_DIR}/out_${RUN_PREFIX}_nsteps_${NSTEPS}.txt

