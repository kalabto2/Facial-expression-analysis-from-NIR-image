#! /bin/bash

# Get the directory containing the script
script_dir=$(dirname "$(readlink -f "$0")")
# Get the parent directory - that is the PROJECT_HOME
PROJECT_HOME=$(dirname "$script_dir")/

echo "Path to the script: $script_dir"
echo "PROJECT_HOME directory: $PROJECT_HOME"

# ================= PARAMETERS =================
use_gpu=1
mode='test-onnx'

# ------------------ MODEL ---------------------
model_checkpoint_fp=$PROJECT_HOME'experiments/logs/CycleGAN_model_logger/version_2023_11_07___17_40_49/checkpoints/25-224640.ckpt'
model_hparams_fp=$PROJECT_HOME'experiments/logs/CycleGAN_model_logger/version_2023_11_07___17_40_49/events.out.tfevents.1699378852.03906a72ec88.70480.0'

# ------------------- TEST ---------------------
test_split_fp=$PROJECT_HOME'splits/preproc_test_split_E.json'

# ------------------- ONNX ---------------------
onnx_fp=$PROJECT_HOME'models/cyclegan-1-25epochs-cycle10.onnx'

# ==============================================

python3 $PROJECT_HOME./gan_cli_test.py --model_checkpoint_fp $model_checkpoint_fp \
                                --model_hparams_fp $model_hparams_fp \
                                --test_split_fp $test_split_fp \
                                --use_gpu $use_gpu \
                                --mode $mode \
                                --onnx_fp $onnx_fp