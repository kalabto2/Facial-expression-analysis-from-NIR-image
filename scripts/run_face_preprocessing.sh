#!/bin/bash

# Get the directory containing the script
script_dir=$(dirname "$(readlink -f "$0")")
# Get the parent directory - that is the PROJECT_HOME
PROJECT_HOME=$(dirname "$script_dir")/

echo "Path to the script: $script_dir"
echo "PROJECT_HOME directory: $PROJECT_HOME"


SPLIT_VERSION="D"
VERSION="F"

# ================= SPLIT PATHS =================
train_split_pth=$PROJECT_HOME"splits/train_split_"$SPLIT_VERSION".json"
test_split_pth=$PROJECT_HOME"splits/test_split_"$SPLIT_VERSION".json"
val_split_pth=$PROJECT_HOME"splits/val_split_"$SPLIT_VERSION".json"

# ================= NEW DATA PATHS =================
new_train_vl_pth=$PROJECT_HOME"data/"$VERSION"_PreprocImg/VL/Strong/train"
new_train_ni_pth=$PROJECT_HOME"data/"$VERSION"_PreprocImg/NI/Strong/train"
new_test_vl_pth=$PROJECT_HOME"data/"$VERSION"_PreprocImg/VL/Strong/test"
new_test_ni_pth=$PROJECT_HOME"data/"$VERSION"_PreprocImg/NI/Strong/test"
new_val_vl_pth=$PROJECT_HOME"data/"$VERSION"_PreprocImg/VL/Strong/val"
new_val_ni_pth=$PROJECT_HOME"data/"$VERSION"_PreprocImg/NI/Strong/val"

# ================= NEW SPLIT PATHS =================
new_train_split_pth=$PROJECT_HOME"splits/preproc_train_split_"$VERSION".json"
new_test_split_pth=$PROJECT_HOME"splits/preproc_test_split_"$VERSION".json"
new_val_split_pth=$PROJECT_HOME"splits/preproc_val_split_"$VERSION".json"

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
                                              --target_size 256 256