#!/bin/bash

train_split_pth="../splits/train_split.json"
test_split_pth="../splits/test_split.json"
new_train_vl_pth="../data/G_PreprocImg/VL/Strong/train"
new_train_ni_pth="../data/G_PreprocImg/NI/Strong/train"
new_test_vl_pth="../data/G_PreprocImg/VL/Strong/test"
new_test_ni_pth="../data/G_PreprocImg/NI/Strong/test"
detector_backend="mtcnn"

python3 ../skeleton/data/face_preprocessing.py --train_split_pth $train_split_pth \
                                              --test_split_pth $test_split_pth \
                                              --new_train_vl_pth $new_train_vl_pth \
                                              --new_train_ni_pth $new_train_ni_pth \
                                              --new_test_vl_pth $new_test_vl_pth \
                                              --new_test_ni_pth $new_test_ni_pth \
                                              --detector_backend $detector_backend