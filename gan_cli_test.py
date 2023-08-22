import pathlib
import click

from lightning.pytorch import Trainer

from skeleton.data.oulu_casia import OuluCasiaDataModule
from skeleton.models.CycleGAN import CycleGAN


@click.command()
@click.option("--model_checkpoint_fp", type=pathlib.Path, help="TBD")
@click.option("--model_hparams_fp", type=pathlib.Path, help="TBD")
@click.option("--test_split_fp", type=pathlib.Path, help="TBD")
def main(
    model_checkpoint_fp: pathlib.Path,
    model_hparams_fp: pathlib.Path,
    test_split_fp: pathlib.Path,
    use_gpu: bool
):
    model = CycleGAN.load_from_checkpoint(
        checkpoint_path=model_checkpoint_fp,
        hparams_file=model_hparams_fp,
        map_location=None,
    )

    dm = OuluCasiaDataModule(test_split_fp=test_split_fp)

    tensorboard_logger = TensorBoardLogger(save_dir="logs/")

    # initialize Trainer
    trainer = Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        logger=[tensorboard_logger],
        default_root_dir="logs",
    )

    trainer.test(datamodule=dm, model=model)


if __name__ == "__main__":
    main()
