import pathlib
from datetime import datetime

import click
import torch

from lightning.pytorch import Trainer

from skeleton.data.oulu_casia import OuluCasiaDataModule
from skeleton.models.CycleGAN import CycleGAN

from lightning.pytorch.loggers import TensorBoardLogger


@click.command()
@click.option("--model_checkpoint_fp", type=pathlib.Path, help="TBD")
@click.option("--model_hparams_fp", type=pathlib.Path, help="TBD")
@click.option("--test_split_fp", type=pathlib.Path, help="TBD")
@click.option(
    "--mode",
    type=str,
    default="",
    help="type 'onnx'/'test-onnx'/'test' for only exporting to onnx/testing and exporting to onnx/ only testing",
)
@click.option("--onnx_fp", type=pathlib.Path, default=None)
@click.option("--use_gpu", type=bool, help="TBD")
def main(
    model_checkpoint_fp: pathlib.Path,
    model_hparams_fp: pathlib.Path,
    test_split_fp: pathlib.Path,
    mode: str,
    onnx_fp: pathlib.Path,
    use_gpu: bool,
):

    assert mode in [
        "onnx",
        "test-onnx",
        "test",
    ], "'mode' argument needs to be one of 'onnx', 'test-onnx', 'test'"

    # Print argument names and their values
    local_symbols = locals()
    str_args = ""
    for arg_name, arg_value in local_symbols.items():
        str_args += f"'{arg_name}': '{arg_value}'\n"
    print("=" * 10, "arguments", "=" * 10)
    print(str_args)
    print("=" * 31)

    save_dir = pathlib.Path("experiments", "logs")
    name = "CycleGAN_model_logger-tests"
    version = datetime.now().strftime(list(model_checkpoint_fp.parts)[-3])

    model = CycleGAN.load_from_checkpoint(
        checkpoint_path=model_checkpoint_fp,
        hparams_file=model_hparams_fp,
        map_location=None,
    )

    dm = OuluCasiaDataModule(test_split_fp=test_split_fp)

    if mode in ["test-onnx", "test"]:
        tensorboard_logger = TensorBoardLogger(
            save_dir=str(save_dir),
            version=version,
            name=name,
        )

        # initialize Trainer
        trainer = Trainer(
            accelerator="gpu" if use_gpu else "cpu",
            devices=1,
            logger=[tensorboard_logger],
            default_root_dir="logs",
        )

        trainer.test(datamodule=dm, model=model)

    if mode in ["test-onnx", "onnx"]:
        model.to_onnx(onnx_fp, torch.randn((3, *dm.image_shape)), export_params=True)


if __name__ == "__main__":
    main()
