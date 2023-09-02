#!/bin/bash

vl_data_path="../data/B_OriginalImg/VL/Strong"  # data to be split
ni_data_path="../data/B_OriginalImg/NI/Strong"  # data to be split

json_train_split_pth="../splits/train_split.json"
json_test_split_pth="../splits/test_split.json"
json_val_split_pth="../splits/val_split.json"

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