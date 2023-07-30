import gc
import os
import re
import time
from typing import List, Set, Optional

import lightning.pytorch as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class OuluCasiaDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "VL",
                 transform: Optional[transforms.Compose] = None,
                 lights: Optional[Set[str]] = "Strong", on_the_fly: bool = True):
        start = time.time()

        assert split == "VL" or split == "NI", f"invalid split value: {split}"
        assert len(lights - {"Strong", "Dark", "Weak"}) == 0, f"invalid light value in {lights}"

        self.root_dir: str = root_dir
        self.split: str = split
        self.transform: Optional[transforms.Compose] = transform
        self.on_the_fly: bool = on_the_fly
        self.light: Optional[Set[str]] = lights

        # retrieve VL data filepaths
        self.data_filepaths: List[str] = []
        file_pattern = r"\d{3}\.jpeg"
        for light in lights:
            for root, directories, filenames in os.walk(os.path.join(self.root_dir, split, light)):
                matching_filepaths = [os.path.join(root, filename) for filename in filenames if
                                      re.match(file_pattern, filename)]
                self.data_filepaths.extend(matching_filepaths)

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
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 2):
        super().__init__()
        self.image_shape = (240, 320)  # TODO un-hardcode?
        self.ni_dataset = None
        self.vl_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # TODO Add more transformations as needed
        ])

        # TODO get rid of hard fixed data - of light types?
        self.vl_dataset = OuluCasiaDataset(self.data_dir, split="VL", transform=transform, lights={"Strong"})
        self.ni_dataset = OuluCasiaDataset(self.data_dir, split="NI", transform=transform, lights={"Strong"})

    def train_dataloader(self):
        vl_dataloader = DataLoader(self.vl_dataset, batch_size=self.batch_size, shuffle=True,
                                   num_workers=self.num_workers)
        ni_dataloader = DataLoader(self.ni_dataset, batch_size=self.batch_size, shuffle=True,
                                   num_workers=self.num_workers)
        return vl_dataloader, ni_dataloader
