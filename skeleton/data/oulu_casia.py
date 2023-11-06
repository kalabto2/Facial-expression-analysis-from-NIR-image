import gc
import json
import os
import pathlib
import re
import time
from typing import List, Set, Optional

import lightning.pytorch as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch


class OuluCasiaDataset(Dataset):
    def __init__(
        self,
        split: str = "both",
        transform: Optional[transforms.Compose] = None,
        on_the_fly: bool = True,
        from_vl_split_fp: Optional[List[pathlib.Path]] = None,
        from_ni_split_fp: Optional[List[pathlib.Path]] = None,
    ):
        start = time.time()

        # asserts
        assert split in ["VL", "NI", "both"], f"invalid split value: {split}"

        self.split: str = split
        self.transform: Optional[transforms.Compose] = transform
        self.on_the_fly: bool = on_the_fly
        self.from_vl_split_fp: Optional[List[pathlib.Path]] = from_vl_split_fp
        self.from_ni_split_fp: Optional[List[pathlib.Path]] = from_ni_split_fp

        if not on_the_fly:
            raise NotImplementedError
            # self.data = [
            #     Image.open(filepath).convert("RGB" if self.split == "VL" else "L")
            #     for filepath in self.data_filepaths
            # ]
            #
            # if self.transform is not None:
            #     self.data = [self.transform(i) for i in self.data]
            #
            # print(gc.collect())

        print("Elapsed OuluCasiaDataset init time:", time.time() - start)

    def __len__(self):
        # Return the total number of samples in the dataset
        return (
            len(self.from_ni_split_fp)
            if self.split == "NI"
            else len(self.from_vl_split_fp)
        )

    def __getitem__(self, idx):  # TODO finish
        if not self.on_the_fly:
            raise NotImplementedError
        else:
            image = []
            if self.split in ["both", "VL"]:
                filepath = self.from_vl_split_fp[idx]
                vl_image = Image.open(filepath).convert("RGB")

                # transforms to Tensor and other
                if self.transform is not None:
                    vl_image = self.transform(vl_image)

                image.append(vl_image)

            if self.split in ["both", "NI"]:
                filepath = self.from_ni_split_fp[idx]
                ni_image = Image.open(filepath).convert("L")

                # transforms to Tensor and other
                if self.transform is not None:
                    ni_image = self.transform(ni_image)

                image.append(ni_image)

        if self.split == "both":
            image[1] = torch.stack([image[1]] * 3, dim=1)[0]
            return torch.stack(image)
        else:
            return image[0]


class OuluCasiaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 2,
        train_split_fp: pathlib.Path = None,
        test_split_fp: pathlib.Path = None,
        val_split_fp: pathlib.Path = None,
        shuffle: bool = True,
    ):
        super().__init__()
        self.image_shape = None
        self.train_vl_dataset = None
        self.train_ni_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_split = None
        self.test_split = None
        self.val_split = None

        # just get the shape of images
        if train_split_fp:
            with open(train_split_fp, "r") as f:
                self.train_split = json.load(f)
                self.image_shape = Image.open(self.train_split["vl"][0]).size
        if test_split_fp:
            with open(test_split_fp, "r") as f:
                self.test_split = json.load(f)
                self.image_shape = Image.open(self.test_split["vl"][0]).size
        if val_split_fp:
            with open(val_split_fp, "r") as f:
                self.val_split = json.load(f)
                self.image_shape = Image.open(self.val_split["vl"][0]).size

    def setup(self, stage=None):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # TODO Add more transformations as needed
            ]
        )
        if self.train_split:
            self.train_vl_dataset = OuluCasiaDataset(
                split="VL", transform=transform, from_vl_split_fp=self.train_split["vl"]
            )
            self.train_ni_dataset = OuluCasiaDataset(
                split="NI", transform=transform, from_ni_split_fp=self.train_split["ni"]
            )

        if self.test_split:
            self.test_dataset = OuluCasiaDataset(
                split="both",
                transform=transform,
                from_vl_split_fp=self.test_split["vl"],
                from_ni_split_fp=self.test_split["ni"],
            )

        if self.val_split:
            self.val_dataset = OuluCasiaDataset(
                split="both",
                transform=transform,
                from_vl_split_fp=self.val_split["vl"],
                from_ni_split_fp=self.val_split["ni"],
            )

    def train_dataloader(self):
        vl_dataloader = DataLoader(
            self.train_vl_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
        ni_dataloader = DataLoader(
            self.train_ni_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
        return vl_dataloader, ni_dataloader

    def test_dataloader(self):
        if not self.test_split:
            return

        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return dataloader
