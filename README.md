# Facial Expression Analysis from NIR image

***

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