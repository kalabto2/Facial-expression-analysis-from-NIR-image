#!/bin/bash

# Get the directory containing the script
script_dir=$(dirname "$(readlink -f "$0")")
# Get the parent directory - that is the PROJECT_HOME
PROJECT_HOME=$(dirname "$script_dir")/

echo "Path to the script: $script_dir"
echo "PROJECT_HOME directory: $PROJECT_HOME"

source $PROJECT_HOME"venv/bin/activate"

# ================= PARAMETERS =================
log_nth_image=1

# ----------------  Data -----------------------
batch_size=1
shuffle_data=0
train_split_fp=$PROJECT_HOME"splits/preproc_train_split_F.json"
val_split_fp=$PROJECT_HOME"splits/preproc_val_split_F.json"
test_split_fp=$PROJECT_HOME"splits/preproc_test_split_F.json"  # '/' stands for no testing, else specify path

# ---------------- Training --------------------
epochs=1
restore_training_from_checkpoint="/" # '/' stands for no restore, else specify path
num_workers=4
use_gpu=0
random_seed=1337

# ------------- Architecture -------------------
weights_init_std=0.02
l_disc=1
l_color=0.0004
l_pix=40
l_feature=1.3

# ------------- Optimization -------------------
learning_rate=2e-4
beta1=0.5
#scheduler_enabled=0
#scheduler_step_freq=1
#scheduler_n_steps=8
#scheduler_eta_min=2e-5
# ==============================================

python3 ./denseUnetGan_cli.py --batch_size $batch_size \
                    --learning_rate $learning_rate \
                    --epochs $epochs \
                    --use_gpu $use_gpu \
                    --random_seed $random_seed \
                    --num_workers $num_workers \
                    --beta1 $beta1 \
                    --log_nth_image $log_nth_image \
                    --restore_training_from_checkpoint $restore_training_from_checkpoint \
                    --weights_init_std $weights_init_std \
                    --shuffle_data $shuffle_data \
                    --train_split_fp $train_split_fp \
                    --val_split_fp $val_split_fp \
                    --test_split_fp $test_split_fp \
                    --l_disc $l_disc \
                    --l_color $l_color \
                    --l_pix $l_pix \
                    --l_feature $l_feature
#                    --scheduler_enabled $scheduler_enabled \
#                    --scheduler_step_freq $scheduler_step_freq \
#                    --scheduler_n_steps $scheduler_n_steps \
#                    --scheduler_eta_min $scheduler_eta_min
