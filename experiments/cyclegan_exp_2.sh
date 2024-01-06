#! /bin/bash

# Get the directory containing the script
script_dir=$(dirname "$(readlink -f "$0")")
# Get the parent directory - that is the PROJECT_HOME
PROJECT_HOME=$(dirname "$script_dir")/

echo "Path to the script: $script_dir"
echo "PROJECT_HOME directory: $PROJECT_HOME"

# ================= PARAMETERS =================
log_nth_image=100

# ----------------  Data -----------------------
batch_size=1
shuffle_data=1
train_split_fp=$PROJECT_HOME"splits/preproc_train_split_E.json"
val_split_fp=$PROJECT_HOME"splits/preproc_val_split_E.json"

# ---------------- Training --------------------
epochs=250
restore_training_from_checkpoint="/" # '/' stands for no restore, else specify path
num_workers=4
use_gpu=1
random_seed=1337
num_generators_optimization=2

# ------------- Architecture -------------------
n_residual_blocks=9
lambda_idt=0
lambda_cycle=6
lambda_discriminator=1

# ------------- Optimization -------------------
train_optim="Adam"
learning_rate=2e-4
beta1=0.5
scheduler_enabled=1
scheduler="linear"
linear_lr_w_init_lr=80,
linear_lr_w_decay=80,
scheduler_step_freq=40
scheduler_n_steps=1000
scheduler_eta_min=2e-5

# ------------- Initialization -----------------
weights_init='kaiming'
weights_init_std=0.02

# ==============================================

python3 ./gan_cli_train.py --batch_size $batch_size \
                    --learning_rate $learning_rate \
                    --train_optim $train_optim \
                    --epochs $epochs \
                    --use_gpu $use_gpu \
                    --random_seed $random_seed \
                    --num_workers $num_workers \
                    --n_residual_blocks $n_residual_blocks \
                    --beta1 $beta1 \
                    --lambda_idt $lambda_idt \
                    --lambda_cycle $lambda_cycle \
                    --lambda_discriminator $lambda_discriminator \
                    --log_nth_image $log_nth_image \
                    --restore_training_from_checkpoint $restore_training_from_checkpoint \
                    --scheduler_enabled $scheduler_enabled \
                    --scheduler_step_freq $scheduler_step_freq \
                    --scheduler_n_steps $scheduler_n_steps \
                    --scheduler_eta_min $scheduler_eta_min \
                    --shuffle_data $shuffle_data \
                    --train_split_fp $train_split_fp \
                    --val_split_fp $val_split_fp \
                    --weights_init $weights_init \
                    --weights_init_std $weights_init_std \
                    --num_generators_optimization $num_generators_optimization \
                    --scheduler $scheduler \
                    --linear_lr_w_decay $linear_lr_w_decay \
                    --linear_lr_w_decay $linear_lr_w_decay
