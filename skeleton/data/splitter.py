import pathlib
import random
import json

import click
import os


class DatasetSplitter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # prepare paths
        os.makedirs(kwargs["json_train_split_pth"].parent, exist_ok=True)
        os.makedirs(kwargs["json_test_split_pth"].parent, exist_ok=True)

        # splits
        self.train_split = {"vl": [], "ni": []}
        self.test_split = {"vl": [], "ni": []}

    def split(self):
        for root, dirs, files in os.walk(self.kwargs["vl_data_path"]):
            # if valid directory - images in it
            if len(files) == 0:
                continue

            # pick sample images for train set
            train_samples = random.sample([os.path.join(root, file) for file in files],
                                          self.kwargs["train_n_img_picked"])
            self.train_split["vl"].extend(train_samples)

            # pick sample images for test set
            rest_samples = [pth for pth in files if os.path.join(root, pth) not in train_samples]
            test_samples = random.sample([os.path.join(root, file) for file in rest_samples],
                                         self.kwargs["test_n_img_picked"])
            self.test_split["vl"].extend(test_samples)

            # get root for files in NI spectra
            ni_root = os.path.join(self.kwargs["ni_data_path"], os.path.join(*list(pathlib.PurePath(root).parts)[-2:]))

            # pick ni train pair images
            ni_train_samples = [os.path.join(ni_root, file) for file in
                                [os.path.split(sample)[1] for sample in train_samples]]
            self.train_split["ni"].extend(ni_train_samples)

            # pick ni test pair images
            ni_test_samples = [os.path.join(ni_root, file) for file in
                               [os.path.split(sample)[1] for sample in test_samples]]
            self.test_split["ni"].extend(ni_test_samples)

    def save_splits(self):
        with open(self.kwargs["json_train_split_pth"], mode="w") as f:
            json.dump(self.train_split, f)
        with open(self.kwargs["json_test_split_pth"], mode="w") as f:
            json.dump(self.test_split, f)

    def __call__(self):
        self.split()
        self.save_splits()


@click.command()
@click.option("--vl_data_path", type=pathlib.Path, help="TBD")
@click.option("--ni_data_path", type=pathlib.Path, help="TBD")
@click.option("--json_train_split_pth", type=pathlib.Path, help="TBD")
@click.option("--json_test_split_pth", type=pathlib.Path, help="TBD")
@click.option("--train_n_img_picked", type=int, default=3, help="TBD")
@click.option("--test_n_img_picked", type=int, default=2, help="TBD")
def main(vl_data_path: pathlib.Path,
         ni_data_path: pathlib.Path,
         train_n_img_picked: int,
         test_n_img_picked: int,
         json_train_split_pth: pathlib.Path,
         json_test_split_pth: pathlib.Path,
         ):
    splitter = DatasetSplitter(vl_data_path=vl_data_path, ni_data_path=ni_data_path,
                               train_n_img_picked=train_n_img_picked, test_n_img_picked=test_n_img_picked,
                               json_train_split_pth=json_train_split_pth, json_test_split_pth=json_test_split_pth)

    splitter()


if __name__ == "__main__":
    main()
