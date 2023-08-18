#!/bin/bash

vl_data_path="../data/B_OriginalImg/VL/Strong"
ni_data_path="../data/B_OriginalImg/NI/Strong"
json_train_split_pth="../splits/train_split.json"
json_test_split_pth="../splits/test_split.json"

python3 ../skeleton/data/splitter.py --vl_data_path $vl_data_path \
                                       --ni_data_path $ni_data_path \
                                       --json_train_split_pth $json_train_split_pth \
                                       --json_test_split_pth $json_test_split_pth