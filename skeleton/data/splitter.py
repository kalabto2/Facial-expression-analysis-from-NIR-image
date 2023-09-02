import pathlib
import random
import json

import click
import os


def folder_to_dict(folder_path):
    folder_dict = {
        "name": os.path.basename(folder_path),
        "type": "folder",
        "contents": [],
    }

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            folder_dict["contents"].append(folder_to_dict(item_path))
        else:
            folder_dict["contents"].append({"name": item, "type": "file"})

    return folder_dict


class DatasetSplitter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # prepare paths
        os.makedirs(kwargs["json_train_split_pth"].parent, exist_ok=True)
        os.makedirs(kwargs["json_test_split_pth"].parent, exist_ok=True)

        # retrieve folder structure
        self.vl_folder = folder_to_dict(self.kwargs["vl_data_path"])
        self.ni_folder = folder_to_dict(self.kwargs["ni_data_path"])

        # splits
        self.train_split = {"vl": [], "ni": []}
        self.test_split = {"vl": [], "ni": []}
        self.val_split = {"vl": [], "ni": []}

    @staticmethod
    def pick_samples(list_of_samples, n_picked):
        # pick sample images for train set
        picked_samples = random.sample(
            list_of_samples,
            n_picked,
        )

        return picked_samples, list(set(list_of_samples) - set(picked_samples))

    def split(self):
        for patient, ni_patient in zip(
            self.vl_folder["contents"], self.ni_folder["contents"]
        ):
            for emotion, ni_emotion in zip(patient["contents"], ni_patient["contents"]):
                # get current images in vl and ni and do its intersection
                vl_images = [
                    sample["name"]
                    for sample in emotion["contents"]
                    if sample["name"][-4:] == "jpeg"
                ]
                ni_images = [
                    sample["name"]
                    for sample in ni_emotion["contents"]
                    if sample["name"][-4:] == "jpeg"
                ]
                intersection = list(set(vl_images) & set(ni_images))

                # pick train samples
                train_samples, intersection_wo_tr = self.pick_samples(
                    intersection, self.kwargs["train_n_img_picked"]
                )

                # pick test samples
                test_samples, intersection_wo_tr_te = self.pick_samples(
                    intersection_wo_tr, self.kwargs["test_n_img_picked"]
                )

                # get paths to current dirs
                vl_pth = os.path.join(
                    self.kwargs["vl_data_path"], patient["name"], emotion["name"]
                )
                ni_pth = os.path.join(
                    self.kwargs["ni_data_path"], patient["name"], emotion["name"]
                )

                # add whole filepaths to collection
                self.train_split["vl"].extend(
                    [os.path.join(vl_pth, fn) for fn in train_samples]
                )
                self.train_split["ni"].extend(
                    [os.path.join(ni_pth, fn) for fn in train_samples]
                )
                self.test_split["vl"].extend(
                    [os.path.join(vl_pth, fn) for fn in test_samples]
                )
                self.test_split["ni"].extend(
                    [os.path.join(ni_pth, fn) for fn in test_samples]
                )

                # there is validation split, split it
                if self.kwargs["val_n_img_picked"] > 0:
                    # pick val samples
                    val_samples, intersection_wo_tr_te_val = self.pick_samples(
                        intersection_wo_tr_te, self.kwargs["val_n_img_picked"]
                    )
                    # add whole filepaths
                    self.val_split["vl"].extend(
                        [os.path.join(vl_pth, fn) for fn in val_samples]
                    )
                    self.val_split["ni"].extend(
                        [os.path.join(ni_pth, fn) for fn in val_samples]
                    )

    def save_splits(self):
        with open(self.kwargs["json_train_split_pth"], mode="w") as f:
            json.dump(self.train_split, f)
        with open(self.kwargs["json_test_split_pth"], mode="w") as f:
            json.dump(self.test_split, f)
        if self.kwargs["val_n_img_picked"] > 0:
            with open(self.kwargs["json_val_split_pth"], mode="w") as f:
                json.dump(self.val_split, f)

    def __call__(self):
        self.split()
        self.save_splits()


@click.command()
@click.option("--vl_data_path", type=pathlib.Path, help="TBD")
@click.option("--ni_data_path", type=pathlib.Path, help="TBD")
@click.option("--json_train_split_pth", type=pathlib.Path, help="TBD")
@click.option("--json_test_split_pth", type=pathlib.Path, help="TBD")
@click.option("--json_val_split_pth", type=pathlib.Path, help="TBD")
@click.option("--train_n_img_picked", type=int, default=3, help="TBD")
@click.option("--test_n_img_picked", type=int, default=2, help="TBD")
@click.option("--val_n_img_picked", type=int, default=0, help="TBD")
def main(
    vl_data_path: pathlib.Path,
    ni_data_path: pathlib.Path,
    train_n_img_picked: int,
    test_n_img_picked: int,
    val_n_img_picked: int,
    json_train_split_pth: pathlib.Path,
    json_test_split_pth: pathlib.Path,
    json_val_split_pth: pathlib.Path,
):
    splitter = DatasetSplitter(
        vl_data_path=vl_data_path,
        ni_data_path=ni_data_path,
        train_n_img_picked=train_n_img_picked,
        test_n_img_picked=test_n_img_picked,
        val_n_img_picked=val_n_img_picked,
        json_train_split_pth=json_train_split_pth,
        json_test_split_pth=json_test_split_pth,
        json_val_split_pth=json_val_split_pth,
    )

    splitter()


if __name__ == "__main__":
    main()
