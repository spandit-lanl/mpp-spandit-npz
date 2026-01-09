#!/bin/bash

#All trainings are run from scratch5

############################################################
# Check that exactly two arguments are provided
############################################################
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <model_size: ti|s|b|l> <mode: pretrain|finetune> <nsteps: 1-16>"

  exit 1
fi

mdl_sz="$1"     # Model Size - Are we using tiny, small, big or large model
mode="$2"         # Training Mode -  Can be pretrain or finetune
ns_str="$3"   # n_steps - Input string for n_steps (history to use)

############################################################
# Validate model size (ti, s, b, l)
############################################################
if [ "$mdl_sz" != "ti" ] && [ "$mdl_sz" != "s" ] && \
   [ "$mdl_sz" != "b" ] && [ "$mdl_sz" != "l" ]; then
  echo "Error: model size must be 'ti', 's', 'b', or 'l'"
  exit 1
fi

############################################################
# Validate prefix mode (pretrain or finetune)
############################################################
if [ "$mode" != "pretrain" ] && [ "$mode" != "finetune" ] && [ "$mode" != "finetune_resume" ] ; then
  echo "Error: mode must be 'pretrain' 'finetune', or 'finetune_resume'"
  exit 1
fi

##################################################
# Validate n_steps range (1..16)
##################################################

[[ "$ns_str" =~ ^[0-9]+$ ]] || { echo "Error: ns must be an integer"; exit 1; }

if [ "$ns_str" -lt 1 ] || [ "$ns_str" -gt 16 ]; then
  echo "Error: ns must be between 1 and 16"
  exit 1
fi

#ns=$(printf "%02d" "$ns_str")  # Format as two digits (01–16)
ns=$(printf "%02d" "$((10#$ns_str))") # Format as two digits (01–16)

############################################################
# Define Training and Fine TUning data
############################################################
train_data='MPP'
finetune_data='LSC'

if [ "$mode" = "pretrain" ]; then
  dataset='MPP'
  cfgname='basic_config'
  runname="pretrain_${mdl_sz}_${dataset}_nsteps_${ns}"
else
  dataset='LSC'
  cfgname="$mode"
  runname="finetune_${mdl_sz}_${dataset}_nsteps_${ns}"
fi

#outdir="./mpp-output/${train_data}/${runname}"
outdir="./mpp-output-finetune/${runname}"

if [ ! -d ${outdir} ]; then
  mkdir -p ${outdir}
fi

cfgfile="config_spandit/mpp_avit_${mdl_sz}_config_nsteps_${ns}.yaml"
if [[ ! -f "$cfgfile" ]]; then
  echo "Error: YAML config not found: $cfgfile"
  exit 1
fi

if [[ ! -r "$cfgfile" ]]; then
  echo "Error: YAML config exists but is not readable: $cfgfile"
  exit 1
fi

############################################################
# Run Training
############################################################

cmd="python train_basic.py --run_name ${runname} --config ${cfgname} --yaml_config ${cfgfile}"

echo "Run Name: ${runname}"
echo "Cfg Name: ${cfgname}"
echo "Cfgfile: ${cfgfile}"
echo -e "\n$cmd &>> ${outdir}/out_${runname}.txt \n"

${cmd} &>> ${outdir}/out_${runname}.txt
#-- use_ddp                \
