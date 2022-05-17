#!/bin/bash

echo "GPU id: $1";
for cfg_file in \
    'configs/train/ft/neuray_ft_scan1_selfsup_minloss.yaml' \
    'configs/train/ft/neuray_ft_scan21_selfsup_minloss.yaml' \
    'configs/train/ft/neuray_ft_scan103_selfsup_minloss.yaml'
do
    echo $cfg_file
    CUDA_VISIBLE_DEVICES=$1 python3 run_training.py --cfg $cfg_file
done