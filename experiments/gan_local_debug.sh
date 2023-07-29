#! /bin/bash

source ../venv/bin/activate

# ================= PARAMETERS =================
log_nth_image=5

# ----------------  Data -----------------------
data_folder="../data/B_OriginalImg"
batch_size=1

# ---------------- Training --------------------
epochs=30
restore_training_from_checkpoint="/" # '/' stands for no restore, else specify path
num_workers=4
use_gpu=0
random_seed=1337

# ------------- Architecture -------------------
n_residual_blocks=6
lambda_idt=0
lambda_cycle=4

# ------------- Optimization -------------------
train_optim="Adam"
learning_rate=2e-4
beta1=0.5
scheduler_enabled=1
scheduler_step_freq=1
scheduler_n_steps=8
scheduler_eta_min=2e-5
# ==============================================

../gan_cli_train.py --data_folder $data_folder \
                    --batch_size $batch_size \
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
                    --log_nth_image $log_nth_image \
                    --restore_training_from_checkpoint $restore_training_from_checkpoint \
                    --scheduler_enabled $scheduler_enabled \
                    --scheduler_step_freq $scheduler_step_freq \
                    --scheduler_n_steps $scheduler_n_steps \
                    --scheduler_eta_min $scheduler_eta_min
