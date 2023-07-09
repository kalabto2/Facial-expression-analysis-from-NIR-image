#! /bin/bash

#lr = 0.0002
#beta1 = 0.5
#lambda_idt = 5.0
#lambda_cycle = 10.0
#sample_interval = 200
#discriminator_freq = 1
#generator_freq = 1

#source ../venv/bin/activate

../gan_cli_train.py --data_folder ../data/B_OriginalImg \
                    --use_gpu 0
