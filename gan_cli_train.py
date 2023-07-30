#!/usr/bin/env python3
import pathlib
import multiprocessing
import click
from datetime import datetime

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch import seed_everything, Trainer

from skeleton.data.oulu_casia import OuluCasiaDataModule
from skeleton.models.CycleGAN import CycleGAN


########################################################################################
# entrypoint of script


@click.command()
@click.option("--data_folder", type=pathlib.Path, required=True, help="path to folder containing data with images")
@click.option("--batch_size", type=int, default=1, help="batch size to use for train and val split")
@click.option("--learning_rate", type=float, default=2e-4, help="constant learning rate used during training")
@click.option("--train_optim", type=str, default='Adam', help="optimizer used during training")
@click.option("--epochs", type=int, default=30, help="number of epochs to train for")
@click.option("--use_gpu", type=bool, default=True, help="whether to use a gpu")
@click.option("--random_seed", type=int, default=1337, help="the random seed")
@click.option("--num_workers", type=int, default=multiprocessing.cpu_count(), help="number of workers to use for data "
                                                                                   "loading")
@click.option("--n_residual_blocks", type=int, default=6, help="Number of residual blocks in both generators")
@click.option("--beta1", type=float, default=0.5, help="beta1 for Adam")
@click.option("--lambda_idt", type=float, default=0.5, help="lambda identity parameter for identity loss")
@click.option("--lambda_cycle", type=float, default=10, help="lambda cycle parameter for cycle loss")
@click.option("--log_nth_image", type=int, default=100, help="Log every nth image of training")
@click.option("--restore_training_from_checkpoint", type=str, default="/", help="TBD")
@click.option("--scheduler_enabled", type=bool, default=False, help="TBD")
@click.option("--scheduler_step_freq", type=int, default=10, help="TBD")
@click.option("--scheduler_n_steps", type=int, default=100, help="TBD")
@click.option("--scheduler_eta_min", type=float, default=2e-5, help="TBD")
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
        lambda_idt: float,
        lambda_cycle: float,
        log_nth_image: int,
        restore_training_from_checkpoint: str,
        scheduler_enabled: bool,
        scheduler_step_freq: int,
        scheduler_n_steps: int,
        scheduler_eta_min: float
):
    # log input
    print("### input arguments ###")
    print(f"batch_size={batch_size}")
    print(f"learning_rate={learning_rate}")
    print(f"epochs={epochs}")
    print(f"use_gpu={use_gpu}")
    print(f"random_seed={random_seed}")
    print(f"num_workers={num_workers}")
    print(f"train_optim={train_optim}")
    print(f"n_residual_blocks={n_residual_blocks}")
    print(f"beta1={beta1}")
    print(f"lambda_idt={lambda_idt}")
    print(f"lambda_cycle={lambda_cycle}")
    print(f"log_nth_image={log_nth_image}")
    print(f"restore_training_from_checkpoint={restore_training_from_checkpoint}")
    print(f"scheduler_enabled={scheduler_enabled}")
    print(f"scheduler_step_freq={scheduler_step_freq}")
    print(f"scheduler_n_steps={scheduler_n_steps}")
    print(f"scheduler_eta_min={scheduler_eta_min}")

    # set random seed
    seed_everything(random_seed)

    # build data loader module
    dm = OuluCasiaDataModule(str(data_folder), batch_size, num_workers=num_workers)

    # build model
    cycle_gan = CycleGAN(3, 1, n_residual_blocks=n_residual_blocks,
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
                         )

    checkpointer = ModelCheckpoint(auto_insert_metric_name=False)

    # initialize Logger
    version = datetime.now().strftime("version_%Y_%m_%d___%H_%M_%S")
    tensorboard_logger = TensorBoardLogger(save_dir="logs/", version=version, name="CycleGAN_model_logger")
    # csv_logger = CSVLogger(save_dir="logs/", version=version) # TODO what with this logger?

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
    trainer.fit(cycle_gan, dm,
                ckpt_path=restore_training_from_checkpoint if restore_training_from_checkpoint != "/" else None)


if __name__ == "__main__":
    main()
