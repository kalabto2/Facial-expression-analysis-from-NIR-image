#! /bin/bash

# Compared to experiment 6 changed:

# Get the directory containing the script
script_dir=$(dirname "$(readlink -f "$0")")
# Get the parent directory - that is the PROJECT_HOME
PROJECT_HOME=$(dirname "$script_dir")/

echo "Path to the script: $script_dir"
echo "PROJECT_HOME directory: $PROJECT_HOME"

# ================= PARAMETERS =================
use_gpu=0
# ------------------ MODEL ---------------------
model_checkpoint_fp=$PROJECT_HOME'/media/tomas/linux_disk_exten/DIP/Facial-expression-analysis-from-NIR-image/logs/CycleGAN_model_logger/version_2023_09_12___12_31_14/checkpoints/7-64.ckpt'
model_hparams_fp=$PROJECT_HOME'/media/tomas/linux_disk_exten/DIP/Facial-expression-analysis-from-NIR-image/logs/CycleGAN_model_logger/version_2023_09_12___12_31_14/hparams.yaml'

# ------------------- TEST ---------------------
test_split_fp=$PROJECT_HOME'split/preproc_test_split_E.json'

# ==============================================

python3 $PROJECT_HOME./gan_cli_test.py --model_checkpoint_fp $model_checkpoint_fp \
                                --model_hparams_fp $model_hparams_fp \
                                --test_split_fp $test_split_fp \
                                --use_gpu $use_gpu