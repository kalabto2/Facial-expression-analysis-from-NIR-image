import json
import pathlib
import os
import click
import cv2

from deepface import DeepFace


class FacePreprocessor:
    def __init__(
        self,
        train_split_pth,
        test_split_pth,
        val_split_pth,
        new_train_vl_pth,
        new_train_ni_pth,
        new_test_vl_pth,
        new_test_ni_pth,
        new_val_ni_pth,
        new_val_vl_pth,
        detector_backend,
        target_size,
        new_train_split_pth=None,
        new_test_split_pth=None,
        new_val_split_pth=None,
    ):
        self.train_split_pth = train_split_pth
        self.test_split_pth = test_split_pth
        self.val_split_pth = val_split_pth
        self.new_train_vl_pth = new_train_vl_pth
        self.new_train_ni_pth = new_train_ni_pth
        self.new_test_vl_pth = new_test_vl_pth
        self.new_test_ni_pth = new_test_ni_pth
        self.new_val_ni_pth = new_val_ni_pth
        self.new_val_vl_pth = new_val_vl_pth
        self.detector_backend = detector_backend
        self.target_size = target_size
        self.new_train_split = new_train_split_pth
        self.new_test_split = new_test_split_pth
        self.new_val_split = new_val_split_pth

    def detect_and_align_face(self, image_fp):
        try:
            face_objs = DeepFace.extract_faces(
                img_path=image_fp,
                target_size=self.target_size,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )
        except Exception as e:
            print(f"ERROR at {image_fp}", e)
            return None

        if len(face_objs) != 1:
            print("NOT FOUND OR MULTIPLE FACES!")
            return None

        face = face_objs[0]["face"]

        return face

    def preprocess_part(self, fps, target_fp, spectra):
        # prepare filepath
        os.makedirs(target_fp, exist_ok=True)

        # align faces for all images
        i = 0
        preprocessed_fps = []
        for fp in fps:
            new_filename = "-".join(pathlib.PurePath(fp).parts[-3:])
            target_path = os.path.join(target_fp, new_filename)

            aligned_face = self.detect_and_align_face(fp)

            if aligned_face is None:
                continue

            aligned_face = 255 * aligned_face[:, :, ::-1]

            cv2.imwrite(target_path, aligned_face)

            print(f"#{i} {spectra} Stored: {new_filename}")
            i += 1
            preprocessed_fps.append(target_path)

        return preprocessed_fps

    def preprocess_split(self, split_pth, new_vl_path, new_ni_pth):
        with open(split_pth, "r") as f:
            paths = json.load(f)

        vl_preproc_fps = self.preprocess_part(paths["vl"], new_vl_path, "vl")
        ni_preproc_fps = self.preprocess_part(paths["ni"], new_ni_pth, "ni")

        return {"vl": vl_preproc_fps, "ni": ni_preproc_fps}

    def preprocess(self):
        # preprocess train split
        train_fps = self.preprocess_split(
            self.train_split_pth, self.new_train_vl_pth, self.new_train_ni_pth
        )

        if self.new_train_split:
            with open(self.new_train_split, "w") as f:
                json.dump(train_fps, f)

        # preprocess test split
        test_fps = self.preprocess_split(
            self.test_split_pth, self.new_test_vl_pth, self.new_test_ni_pth
        )

        if self.new_test_split:
            with open(self.new_test_split, "w") as f:
                json.dump(test_fps, f)

        # preprocess val split
        val_fps = self.preprocess_split(
            self.val_split_pth, self.new_val_vl_pth, self.new_val_ni_pth
        )

        if self.new_val_split:
            with open(self.new_val_split, "w") as f:
                json.dump(val_fps, f)


@click.command()
@click.option("--train_split_pth", type=pathlib.Path, help="TBD")
@click.option("--test_split_pth", type=pathlib.Path, help="TBD")
@click.option("--val_split_pth", type=pathlib.Path, help="TBD")
@click.option("--new_train_vl_pth", type=pathlib.Path, help="TBD")
@click.option("--new_train_ni_pth", type=pathlib.Path, help="TBD")
@click.option("--new_test_vl_pth", type=pathlib.Path, help="TBD")
@click.option("--new_test_ni_pth", type=pathlib.Path, help="TBD")
@click.option("--new_val_vl_pth", type=pathlib.Path, help="TBD")
@click.option("--new_val_ni_pth", type=pathlib.Path, help="TBD")
@click.option("--new_train_split_pth", type=pathlib.Path, default=None, help="TBD")
@click.option("--new_test_split_pth", type=pathlib.Path, default=None, help="TBD")
@click.option("--new_val_split_pth", type=pathlib.Path, default=None, help="TBD")
@click.option("--detector_backend", type=str, default="mtcnn", help="TBD")
@click.option("--target_size", nargs=2, type=click.Tuple([int, int]))
def main(
    train_split_pth: pathlib.Path,
    test_split_pth: pathlib.Path,
    val_split_pth: pathlib.Path,
    new_train_vl_pth: pathlib.Path,
    new_train_ni_pth: pathlib.Path,
    new_test_vl_pth: pathlib.Path,
    new_test_ni_pth: pathlib.Path,
    new_val_vl_pth: pathlib.Path,
    new_val_ni_pth: pathlib.Path,
    detector_backend: str,
    new_train_split_pth: pathlib.Path,
    new_test_split_pth: pathlib.Path,
    new_val_split_pth: pathlib.Path,
    target_size: tuple,
):
    # Print argument names and their values
    local_symbols = locals()
    str_args = ""
    for arg_name, arg_value in local_symbols.items():
        str_args += f"'{arg_name}': {arg_value}'\n"
    print(str_args)

    preprocessor = FacePreprocessor(
        train_split_pth=train_split_pth,
        test_split_pth=test_split_pth,
        val_split_pth=val_split_pth,
        new_train_vl_pth=new_train_vl_pth,
        new_train_ni_pth=new_train_ni_pth,
        new_test_vl_pth=new_test_vl_pth,
        new_test_ni_pth=new_test_ni_pth,
        new_val_vl_pth=new_val_vl_pth,
        new_val_ni_pth=new_val_ni_pth,
        detector_backend=detector_backend,
        target_size=target_size,
        new_train_split_pth=new_train_split_pth,
        new_test_split_pth=new_test_split_pth,
        new_val_split_pth=new_val_split_pth,
    )

    preprocessor.preprocess()


if __name__ == "__main__":
    main()
