#!/bin/bash

# Get the directory containing the script
script_dir=$(dirname "$(readlink -f "$0")")
# Get the parent directory - that is the PROJECT_HOME
PROJECT_HOME=$(dirname "$script_dir")/

echo "Path to the script: $script_dir"
echo "PROJECT_HOME directory: $PROJECT_HOME"


# ================= SPLIT PATHS =================
train_split_pth=$PROJECT_HOME"splits/train_split_D.json"
test_split_pth=$PROJECT_HOME"splits/test_split_D.json"
val_split_pth=$PROJECT_HOME"splits/val_split_D.json"

# ================= NEW DATA PATHS =================
new_train_vl_pth=$PROJECT_HOME"data/D_PreprocImg/VL/Strong/train"
new_train_ni_pth=$PROJECT_HOME"data/D_PreprocImg/NI/Strong/train"
new_test_vl_pth=$PROJECT_HOME"data/D_PreprocImg/VL/Strong/test"
new_test_ni_pth=$PROJECT_HOME"data/D_PreprocImg/NI/Strong/test"
new_val_vl_pth=$PROJECT_HOME"data/D_PreprocImg/VL/Strong/val"
new_val_ni_pth=$PROJECT_HOME"data/D_PreprocImg/NI/Strong/val"

# ================= NEW SPLIT PATHS =================
new_train_split_pth=$PROJECT_HOME"splits/preproc_train_split_D.json"
new_test_split_pth=$PROJECT_HOME"splits/preproc_test_split_D.json"
new_val_split_pth=$PROJECT_HOME"splits/preproc_val_split_D.json"

# ================= OTHER =================
detector_backend="retinaface"

python3 $PROJECT_HOME'skeleton/data/face_preprocessing.py' --train_split_pth $train_split_pth \
                                              --test_split_pth $test_split_pth \
                                              --val_split_pth $val_split_pth \
                                              --new_train_vl_pth $new_train_vl_pth \
                                              --new_train_ni_pth $new_train_ni_pth \
                                              --new_test_vl_pth $new_test_vl_pth \
                                              --new_test_ni_pth $new_test_ni_pth \
                                              --new_val_vl_pth $new_val_vl_pth \
                                              --new_val_ni_pth $new_val_ni_pth \
                                              --detector_backend $detector_backend \
                                              --new_train_split_pth $new_train_split_pth \
                                              --new_test_split_pth $new_test_split_pth \
                                              --new_val_split_pth $new_val_split_pth \
                                              --target_size 150 150