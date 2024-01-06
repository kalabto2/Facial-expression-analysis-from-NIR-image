#!/bin/bash

# Get the directory containing the script
script_dir=$(dirname "$(readlink -f "$0")")
# Get the parent directory - that is the PROJECT_HOME
PROJECT_HOME=$(dirname "$script_dir")/

echo "Path to the script: $script_dir"
echo "PROJECT_HOME directory: $PROJECT_HOME"

vl_data_path=$PROJECT_HOME"data/B_OriginalImg/VL/Strong"  # data to be split
ni_data_path=$PROJECT_HOME"data/B_OriginalImg/NI/Strong"  # data to be split

json_train_split_pth=$PROJECT_HOME"splits/train_split_G.json"
json_test_split_pth=$PROJECT_HOME"splits/test_split_G.json"
json_val_split_pth=$PROJECT_HOME"splits/val_split_G.json"

train_n_img_picked=3
test_n_img_picked=2
val_n_img_picked=2  # for no val_split pass 0

python3 ../skeleton/data/splitter.py   --vl_data_path $vl_data_path \
                                       --ni_data_path $ni_data_path \
                                       --json_train_split_pth $json_train_split_pth \
                                       --json_test_split_pth $json_test_split_pth \
                                       --json_val_split_pth $json_val_split_pth \
                                       --train_n_img_picked $train_n_img_picked \
                                       --test_n_img_picked $test_n_img_picked \
                                       --val_n_img_picked $val_n_img_picked