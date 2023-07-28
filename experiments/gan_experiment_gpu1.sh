#! /bin/bash

../gan_cli_train.py --data_folder ../data/B_OriginalImg \
                    --use_gpu 1 \
                    --num_workers 2 \
                    --log_nth_image 100 \
                    --lambda_idt 0 \
                    --learning_rate 2e-3
