import configparser
import json
from pathlib import Path
from typing import Iterable, Dict


YOLO_ANNOTATIONS_FOLDER_NAME = "YOLO_darknet"
LABELS_FOLDER_NAME = "labels"
PASCAL_VOC_FOLDER_NAME = "PASCAL_VOC"
ORANGE = "orange"
GREEN = "green"
RED = "red"
PURPLE = "purple"


def get_all_jpg_recursive(img_root: Path) -> Iterable[Path]:
    for item in img_root.rglob("*.jpg"):
        yield item


def get_all_txt_recursive(root_dir: Path) -> Iterable[Path]:
    for item in root_dir.rglob("*.txt"):
        yield item


def get_corrected_photo_name(photo_name: Path, expected_num_parts: int, sep: str = "_"):
    photo_ext = photo_name.suffix
    photo_split = photo_name.name.split(sep)
    len_photo_split = len(photo_split)
    if len_photo_split > expected_num_parts:
        photo_name = "_".join(photo_split[0:expected_num_parts])
        photo_name = f"{photo_name}{photo_ext}"
    return photo_name


def get_id_to_label_map(classes_json_path: Path) -> Dict[int, str]:
    """
    Opens a txt file that has one class name per line and assumes
    zero indexed class ids corresponding to the classes as they appear in
    the provided file.

    """
    with open(str(classes_json_path), "r") as json_file:
        data = json.load(json_file)
    label_map = dict()
    for key, val in data.items():
        label_map[int(key)] = val["label"]
    return label_map


def get_config_items(base_dir: Path):
    config = configparser.ConfigParser()
    config_path = base_dir / "config.ini"
    if not config_path.exists():
        raise RuntimeError(f"{str(config_path)} does not exist.")
    config.read(str(config_path))
    python_path = config.get("YOLO", "PYTHON_EXE")
    yolo_root = config.get("YOLO", "BASE_DIR")
    cfg_path = config.get("YOLO", "CFG_PATH")
    weights_path = config.get("YOLO", "WEIGHTS_PATH")
    hyp_path = config.get("YOLO", "HYP_PATH")
    dataset_root = config.get("DATASET", "ROOT")
    classes_json_path = config.get("DATASET", "CLASSES_JSON")
    if (
        classes_json_path is None
        or classes_json_path == ""
        or classes_json_path == "./"
    ):
        classes_json_path = base_dir / "classes.json"
    return (
        python_path,
        yolo_root,
        cfg_path,
        weights_path,
        hyp_path,
        dataset_root,
        classes_json_path,
    )


def get_open_labeling_dir(base_dir: Path = Path(__file__).parents[1]):
    config = configparser.ConfigParser()
    config_path = base_dir / "config.ini"
    if not config_path.exists():
        raise RuntimeError(f"{str(config_path)} does not exist.")
    config.read(str(config_path))
    return config.get("EDITOR", "OPEN_LABELING_ROOT")


def get_version_control_config(base_dir: Path = Path(__file__).parents[1]):
    config = configparser.ConfigParser()
    config_path = base_dir / "config.ini"
    if not config_path.exists():
        raise RuntimeError(f"{str(config_path)} does not exist.")
    config.read(str(config_path))
    git_exe_path = config.get("GIT", "EXE_PATH")
    remote_name = config.get("GIT", "REMOTE_NAME")
    branch_name = config.get("GIT", "BRANCH_NAME")
    return git_exe_path, remote_name, branch_name
