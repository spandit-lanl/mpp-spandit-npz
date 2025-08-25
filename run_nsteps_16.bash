#!/bin/bash

#source ~/venv/mpp/bin/activate

EP500=500


FS4='scratch4'
FS5='scratch5'

DS_FP16_HALF='lsc240420_fp16_half'
DS_FP16_FULL='lsc240420_fp16_full'
DS_FP64_HALF='lsc240420_half'
DS_FP64_FULL='lsc240420'

NOW=`date +"%Y_%m_%d__%H_%M_%S"`



FS=${FS5}
DS=${DS_FP64_FULL}

#All trainings are run from scratch5
python train_basic.py --run_name lsc240420_nsteps_16 --config basic_config --yaml_config config/mpp_lsc_avit_ti_config_nsteps_16.yaml &>>output_lsc240420_nsteps_16.txt
