#!/usr/bin/env python3
import os
import pathlib

from datetime import datetime

import multiprocessing

import click
# import pytorch_lightning
# from pytorch_lightning import LightningModule
# import torch.nn as nn
# from lightning_fabric.plugins.environments import SLURMEnvironment

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch import seed_everything, Trainer

from skeleton.data.oulu_casia import OuluCasiaDataModule
from skeleton.models.CycleGAN import CycleGAN

########################################################################################
# entrypoint of script

@click.command()
@click.option(
    "--data_folder",
    type=pathlib.Path,
    required=True,
    help="path to folder containing data with images",
)
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
    "--train_optim",
    type=str,
    default='Adam',
    help="optimizer used during training",
)
@click.option("--epochs", type=int, default=30, help="number of epochs to train for")
@click.option("--use_gpu", type=bool, default=True, help="whether to use a gpu")
@click.option("--random_seed", type=int, default=1337, help="the random seed")
@click.option("--num_workers", type=int, default=multiprocessing.cpu_count(), help="number of workers to use for data loading")
@click.option("--n_residual_blocks", type=int, default=6, help="Number of residual blocks in both generators")
@click.option("--beta1", type=float, default=0.5, help="beta1 for Adam")
@click.option("--lambda_idt", type=float, default=0.5, help="lambda identity parameter for identity loss")
def main(
    data_folder: pathlib.Path,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    use_gpu: int,
    random_seed: int,
    num_workers: int,
    train_optim: str,
    n_residual_blocks: int,
    beta1: float,
    lambda_idt: float
):
    # log input
    print("### input arguments ###")
    print(f"batch_size={batch_size}")
    print(f"learning_rate={learning_rate}")
    print(f"epochs={epochs}")
    print(f"use_gpu={use_gpu}")
    print(f"random_seed={random_seed}")
    print(f"train_optim={train_optim}")
    print(f"n_residual_blocks={n_residual_blocks}")
    print(f"beta1={beta1}")
    print(f"lambda_idt={lambda_idt}")

    # set random seed
    seed_everything(random_seed)

    # build data loader module
    dm = OuluCasiaDataModule(str(data_folder), batch_size)

    # setup datasets
    dm.setup()

    # retrieve dataloaders
    vl_dataloader, ni_dataloader = dm.train_dataloader()

    # build model
    cycle_gan = CycleGAN(3, 3, n_residual_blocks=n_residual_blocks,
                         lr=learning_rate,
                         beta1=beta1,
                         lambda_idt=lambda_idt)

    # configure callback managing checkpoints, and checkpoint file names
    pattern = "epoch_{epoch:04d}.step_{step:09d}.val-eer_{val_eer:.4f}"
    ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
    checkpointer = ModelCheckpoint(
        monitor="val_eer",
        filename=pattern + ".best",
        save_last=True,
        auto_insert_metric_name=False,
    )

    # initialize trainer
    version = datetime.now().strftime("version_%Y_%m_%d___%H_%M_%S")
    if "SLURM_JOB_ID" in os.environ:
        version += f"___job_id_{os.environ['SLURM_JOB_ID']}"

    tensorboard_logger = TensorBoardLogger(save_dir="logs/", version=version)
    csv_logger = CSVLogger(save_dir="logs/", version=version)

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=[checkpointer, LearningRateMonitor()],
        logger=[tensorboard_logger, csv_logger],
        default_root_dir="logs",
        # plugins=[SLURMEnvironment(auto_requeue=False)],  #
    )

    # train loop
    trainer.fit(cycle_gan, dm)

    # # test loop (on dev set)
    # cycle_gan = cycle_gan.load_from_checkpoint(checkpointer.best_model_path)
    # trainer.test(cycle_gan, datamodule=dm)


if __name__ == "__main__":
    main()
