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


class OuluCasiaDataset(Dataset):
    def __init__(self, split: str = "VL",
                 transform: Optional[transforms.Compose] = None,
                 on_the_fly: bool = True,
                 from_split_fp: Optional[List[pathlib.Path]] = None):
        start = time.time()

        # asserts
        assert split == "VL" or split == "NI", f"invalid split value: {split}"

        self.split: str = split
        self.transform: Optional[transforms.Compose] = transform
        self.on_the_fly: bool = on_the_fly
        self.from_split_fp: Optional[List[pathlib.Path]] = from_split_fp

        # retrieve VL data filepaths
        self.data_filepaths = from_split_fp

        if not on_the_fly:
            self.data = [Image.open(filepath).convert("RGB" if self.split == "VL" else "L")
                         for filepath in self.data_filepaths]

            if self.transform is not None:
                self.data = [self.transform(i) for i in self.data]

            print(gc.collect())

        print("Elapsed OuluCasiaDataset init time:", time.time() - start)

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data_filepaths)

    def __getitem__(self, idx):
        if not self.on_the_fly:
            return self.data[idx]
        else:
            filepath = self.data_filepaths[idx]
            image = Image.open(filepath).convert("RGB" if self.split == "VL" else "L")

        # transforms to Tensor and other
        if self.transform is not None:
            image = self.transform(image)

        return image


class OuluCasiaDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1, num_workers: int = 2, train_split_fp: pathlib.Path = None,
                 test_split_fp: pathlib.Path = None, shuffle: bool = True):
        super().__init__()
        self.image_shape = None  # (240, 320) are the original images
        self.test_vl_dataset = None
        self.train_vl_dataset = None
        self.test_ni_dataset = None
        self.train_ni_dataset = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_split = None
        self.test_split = None
        if train_split_fp:
            with open(train_split_fp, "r") as f:
                self.train_split = json.load(f)
                self.image_shape = Image.open(self.train_split["vl"][0]).size
        if test_split_fp:
            with open(test_split_fp, "r") as f:
                self.test_split = json.load(f)
                self.image_shape = Image.open(self.test_split["vl"][0]).size

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # TODO Add more transformations as needed
        ])
        if self.train_split:
            self.train_vl_dataset = OuluCasiaDataset(split="VL", transform=transform,
                                                     from_split_fp=self.train_split["vl"])
            self.train_ni_dataset = OuluCasiaDataset(split="NI", transform=transform,
                                                     from_split_fp=self.train_split["ni"])

        if self.test_split:
            self.test_vl_dataset = OuluCasiaDataset(split="VL", transform=transform,
                                                    from_split_fp=self.test_split["vl"])
            self.test_ni_dataset = OuluCasiaDataset(split="NI", transform=transform,
                                                    from_split_fp=self.test_split["ni"])

    def train_dataloader(self):
        vl_dataloader = DataLoader(self.train_vl_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                   num_workers=self.num_workers)
        ni_dataloader = DataLoader(self.train_ni_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                   num_workers=self.num_workers)
        return vl_dataloader, ni_dataloader

    def test_dataloader(self):
        if not self.test_split:
            return

        vl_dataloader = DataLoader(self.test_vl_dataset, batch_size=self.batch_size, shuffle=False,
                                   num_workers=self.num_workers)
        ni_dataloader = DataLoader(self.test_ni_dataset, batch_size=self.batch_size, shuffle=False,
                                   num_workers=self.num_workers)
        return vl_dataloader, ni_dataloader
