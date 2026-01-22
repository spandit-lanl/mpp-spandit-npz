#!/usr/bin/env bash
set -euo pipefail

conf_dir='config/sp_lsc'

usage() {
  cat <<'EOF'
Usage: pick_cfg.sh <phase:pretrain|finetune|finetune_resume> <ns:1-16> <lr:3|4> <opt:adam|adan|sgd> <wd:3|4>

Example:
  ./pick_cfg.sh finetune 3 3 T adam 4
EOF
}

if [[ $# -ne 6 ]]; then
  usage
  exit 2
fi

phase="$1"
ns="$2"
lr="$3"
opt="$4"
wd="$5"

# Validate phase
case "$phase" in
  pretrain|finetune|finetune_resume) ;;
  *)
    echo "ERROR: phase must be pretrain, finetune, or finetune_resume (got: $phase)" >&2
    exit 2
    ;;
esac

# Validate ns: integer 1..16
if ! [[ "$ns" =~ ^[0-9]+$ ]] || (( ns < 1 || ns > 16 )); then
  echo "ERROR: ns must be an integer from 1 to 16 (got: $ns)" >&2
  exit 2
fi

# Validate remaining args
case "$lr" in 3|4) ;; *) echo "ERROR: lr must be 3 or 4 (got: $lr)" >&2; exit 2;; esac
case "$opt" in adam|adan|sgd) ;; *) echo "ERROR: opt must be adam, adan, or sgd (got: $opt)" >&2; exit 2;; esac
case "$wd" in 3|4) ;; *) echo "ERROR: wd must be 3 or 4 (got: $wd)" >&2; exit 2;; esac

# Zero-pad ns to 2 digits
ns2="$(printf '%02d' "$ns")"

# Config filename
CFG_NAME="mpp_avit_B_ns-${ns2}_lr-${lr}_opt-${opt}_wd-${wd}.yaml"

# RUN_NAME should use "finetune" for both finetune and finetune_resume
run_phase="$phase"
if [[ "$phase" == "finetune_resume" ]]; then
  run_phase="finetune"
fi

# Construct RUN_NAME by replacing the whole prefix and removing extension
RUN_NAME="${CFG_NAME%.yaml}"
RUN_NAME="${RUN_NAME/mpp_avit_B_ns-${ns2}/${run_phase}_${ns2}}"

# (Optional) verify config exists
if [[ ! -f "${conf_dir}/${CFG_NAME}" ]]; then
  echo "ERROR: Config not found: ${conf_dir}/${CFG_NAME}" >&2
  exit 1
else
  echo "All Good"
fi

# Build the training command
CMD=(python train_basic.py
     --run_name "$RUN_NAME"
     --config "$phase"
     --yaml_config="${conf_dir}/${CFG_NAME}")

echo  "RUN_NAME=$RUN_NAME"
LOG=" &>>mpp-output-expt/out_${RUN_NAME}.log"
cmd=`printf '%q ' "${CMD[@]}"`
cmd="${cmd} &>>${LOG}"
echo "${cmd}"
# Output
#echo "CFG_NAME=$CFG_NAME"
#echo sed "s/pretrain_B_MPP_nsteps_01/${RUN_NAME}/" ${conf_dir}/${CFG_NAME}
#printf 'CMD='; printf '%q ' "${CMD[@]}"; echo
echo
# Uncomment to execute:
# "${CMD[@]}"

