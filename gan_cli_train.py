import multiprocessing
import pathlib
from datetime import datetime

import click
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from skeleton.data.oulu_casia import OuluCasiaDataModule
from skeleton.models.CycleGAN import CycleGAN


########################################################################################
# entrypoint of script


@click.command()
@click.option("--train_split_fp", type=pathlib.Path, default=None, help="TBD")
@click.option("--val_split_fp", type=pathlib.Path, default=None, help="TBD")
@click.option(
    "--batch_size",
    type=int,
    default=1,
    help="batch size to use for train and val split",
)
@click.option(
    "--learning_rate",
    type=float,
    default=2e-4,
    help="constant learning rate used during training",
)
@click.option(
    "--train_optim", type=str, default="Adam", help="optimizer used during training"
)
@click.option("--epochs", type=int, default=30, help="number of epochs to train for")
@click.option("--use_gpu", type=bool, default=True, help="whether to use a gpu")
@click.option("--random_seed", type=int, default=1337, help="the random seed")
@click.option(
    "--num_workers",
    type=int,
    default=multiprocessing.cpu_count(),
    help="number of workers to use for data " "loading",
)
@click.option(
    "--n_residual_blocks",
    type=int,
    default=6,
    help="Number of residual blocks in both generators",
)
@click.option("--beta1", type=float, default=0.5, help="beta1 for Adam")
@click.option(
    "--lambda_idt",
    type=float,
    default=0.5,
    help="lambda identity parameter for identity loss",
)
@click.option(
    "--lambda_cycle",
    type=float,
    default=10,
    help="lambda cycle parameter for cycle loss",
)
@click.option(
    "--log_nth_image", type=int, default=100, help="Log every nth image of training"
)
@click.option("--restore_training_from_checkpoint", type=str, default="/", help="TBD")
@click.option("--scheduler_enabled", type=bool, default=False, help="TBD")
@click.option("--scheduler_step_freq", type=int, default=10, help="TBD")
@click.option("--scheduler_n_steps", type=int, default=100, help="TBD")
@click.option("--scheduler_eta_min", type=float, default=2e-5, help="TBD")
@click.option("--weights_init", type=str, default="normal", help="TBD")
@click.option("--weights_init_std", type=float, default=0.02, help="TBD")
@click.option("--lambda_discriminator", type=float, default=0.5, help="TBD")
@click.option("--scheduler", type=str, default="linear", help="TBD")
@click.option("--linear_lr_w_init_lr", type=int, default=5, help="TBD")
@click.option("--linear_lr_w_decay", type=int, default=5, help="TBD")
@click.option("--shuffle_data", type=bool, default=True, help="TBD")
@click.option("--check_val_every_n_epoch", type=int, default=1, help="TBD")
def main(
    batch_size: int,
    learning_rate: float,
    epochs: int,
    use_gpu: int,
    random_seed: int,
    num_workers: int,
    train_optim: str,
    n_residual_blocks: int,
    beta1: float,
    lambda_idt: float,
    lambda_cycle: float,
    log_nth_image: int,
    restore_training_from_checkpoint: str,
    scheduler_enabled: bool,
    scheduler_step_freq: int,
    scheduler_n_steps: int,
    scheduler_eta_min: float,
    weights_init: str,
    weights_init_std: float,
    lambda_discriminator: float,
    scheduler: str,
    linear_lr_w_init_lr: int,
    linear_lr_w_decay: int,
    shuffle_data: bool,
    train_split_fp: pathlib.Path,
    val_split_fp: pathlib.Path,
    check_val_every_n_epoch: int,
):
    # Print argument names and their values
    local_symbols = locals()
    str_args = ""
    for arg_name, arg_value in local_symbols.items():
        str_args += f"'{arg_name}': '{arg_value}'\n"
    print("=" * 10, "arguments", "=" * 10)
    print(str_args)
    print("=" * 31)

    # set random seed
    seed_everything(random_seed)

    # build data loader module
    dm = OuluCasiaDataModule(
        batch_size,
        num_workers=num_workers,
        train_split_fp=train_split_fp,
        val_split_fp=val_split_fp,
        shuffle=shuffle_data,
    )

    # build model
    cycle_gan = CycleGAN(
        3,
        1,
        n_residual_blocks=n_residual_blocks,
        lr=learning_rate,
        beta1=beta1,
        lambda_idt=lambda_idt,
        lambda_cycle=lambda_cycle,
        image_shape=dm.image_shape,
        log_nth_image=log_nth_image,
        scheduler_step_freq=scheduler_step_freq,
        scheduler_n_steps=scheduler_n_steps,
        scheduler_enabled=scheduler_enabled,
        scheduler_eta_min=scheduler_eta_min,
        weights_init=weights_init,
        weights_init_std=weights_init_std,
        lambda_discriminator=lambda_discriminator,
        scheduler=scheduler,
        linear_lr_w_init_lr=linear_lr_w_init_lr,
        linear_lr_w_decay=linear_lr_w_decay,
    )

    checkpointer = ModelCheckpoint(auto_insert_metric_name=False)

    if restore_training_from_checkpoint == "/":
        # initialize Logger
        version = datetime.now().strftime("version_%Y_%m_%d___%H_%M_%S")
        tensorboard_logger = TensorBoardLogger(
            save_dir="experiments/logs/", version=version, name="CycleGAN_model_logger"
        )
    else:
        # initialize Logger
        version = restore_training_from_checkpoint.split("/")[-2]
        tensorboard_logger = TensorBoardLogger(
            save_dir="experiments/logs/", version=version, name="CycleGAN_model_logger"
        )

    # initialize Trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=[checkpointer, LearningRateMonitor(logging_interval="step")],
        logger=[tensorboard_logger],
        default_root_dir="experiments/logs",
        check_val_every_n_epoch=check_val_every_n_epoch,
    )

    # train loop
    trainer.fit(
        cycle_gan,
        dm,
        ckpt_path=restore_training_from_checkpoint
        if restore_training_from_checkpoint != "/"
        else None,
    )


if __name__ == "__main__":
    main()
