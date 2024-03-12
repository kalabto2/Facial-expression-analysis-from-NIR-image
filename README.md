# Facial Expression Analysis from NIR image

***

> This branch contains code for models DenseUnet and CycleGAN. Although CycleGAN was used in develeopment, this particular code was not, however, it should be working.

## Setting up

Prepare virtual environment.
```bash
virtualenvironment venv
```

If running local, activate virtual environment
```bash
source venv/bin/activate
```

Run experiment, for example *experiments/gan_experiment_gpu1.sh*.
Experiments must be ran from *experiments/* directory.
```bash
cd experiments/
./gan_experiment_gpu1.sh
```

Run *Tensorboard* analytic dashboard.
```bash
tensorboard --logdir=experiments/logs --bind_all  --samples_per_plugin=images=100
```