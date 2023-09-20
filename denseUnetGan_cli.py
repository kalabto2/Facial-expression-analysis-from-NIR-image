#!/usr/bin/env python3
import multiprocessing
import pathlib
from datetime import datetime

import click
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from skeleton.data.oulu_casia import OuluCasiaDataModule
from skeleton.models.DenseUnetGAN import DenseUnetGAN


@click.command()
@click.option("--train_split_fp", type=pathlib.Path, default=None, help="TBD")
@click.option("--val_split_fp", type=pathlib.Path, default=None, help="TBD")
@click.option("--batch_size", type=int, default=1, help="TBD")
@click.option("--learning_rate", type=float, default=2e-4, help="TBD")
@click.option("--epochs", type=int, default=30, help="number of epochs to train for")
@click.option("--use_gpu", type=bool, default=True, help="whether to use a gpu")
@click.option("--random_seed", type=int, default=1337, help="the random seed")
@click.option(
    "--num_workers",
    type=int,
    default=multiprocessing.cpu_count(),
    help="number of workers to use for data " "loading",
)
@click.option("--beta1", type=float, default=0.5, help="beta1 for Adam")
@click.option(
    "--log_nth_image", type=int, default=100, help="Log every nth image of training"
)
@click.option("--restore_training_from_checkpoint", type=str, default="/", help="TBD")
# @click.option("--scheduler_enabled", type=bool, default=False, help="TBD")
# @click.option("--scheduler_step_freq", type=int, default=10, help="TBD")
# @click.option("--scheduler_n_steps", type=int, default=100, help="TBD")
# @click.option("--scheduler_eta_min", type=float, default=2e-5, help="TBD")
@click.option("--weights_init_std", type=float, default=0.02, help="TBD")
# @click.option("--lambda_discriminator", type=float, default=0.5, help="TBD")
# @click.option("--scheduler", type=str, default="linear", help="TBD")
# @click.option("--linear_lr_w_init_lr", type=int, default=5, help="TBD")
# @click.option("--linear_lr_w_decay", type=int, default=5, help="TBD")
@click.option("--shuffle_data", type=bool, default=True, help="TBD")
@click.option("--l_disc", type=float, default=1, help="TBD")
@click.option("--l_color", type=float, default=0.0004, help="TBD")
@click.option("--l_pix", type=float, default=40, help="TBD")
@click.option("--l_feature", type=float, default=1.3, help="TBD")
def main(
    batch_size: int,
    learning_rate: float,
    epochs: int,
    use_gpu: int,
    random_seed: int,
    num_workers: int,
    beta1: float,
    log_nth_image: int,
    restore_training_from_checkpoint: str,
    # scheduler_enabled: bool,
    # scheduler_step_freq: int,
    # scheduler_n_steps: int,
    # scheduler_eta_min: float,
    weights_init_std: float,
    # lambda_discriminator: float,
    # scheduler: str,
    # linear_lr_w_init_lr: int,
    # linear_lr_w_decay: int,
    shuffle_data: bool,
    train_split_fp: pathlib.Path,
    val_split_fp: pathlib.Path,
    l_disc: float,
    l_color: float,
    l_pix: float,
    l_feature: float,
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

    dense_unet = DenseUnetGAN(
        1,
        3,
        dm.image_shape,
        learning_rate,
        beta1,
        weights_init_std,
        l_disc=l_disc,
        l_color=l_color,
        l_pix=l_pix,
        l_feature=l_feature,
    )

    checkpointer = ModelCheckpoint(auto_insert_metric_name=False)

    # initialize Logger
    version = datetime.now().strftime("version_%Y_%m_%d___%H_%M_%S")
    tensorboard_logger = TensorBoardLogger(
        save_dir="experiments/logs/", version=version, name="DenseUnetGAN_model_logger"
    )

    # initialize Trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=[checkpointer, LearningRateMonitor(logging_interval="step")],
        logger=[tensorboard_logger],
        default_root_dir="logs",
    )

    # train loop
    trainer.fit(
        dense_unet,
        dm,
        ckpt_path=restore_training_from_checkpoint
        if restore_training_from_checkpoint != "/"
        else None,
    )


if __name__ == "__main__":
    main()
