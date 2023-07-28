#! /bin/bash

source ../venv/bin/activate

../gan_cli_train.py --data_folder ../data/B_OriginalImg \
                    --use_gpu 0 \
                    --num_workers 4 \
                    --log_nth_image 5 \
                    --lambda_idt 0 \
                    --learning_rate 2e-3

