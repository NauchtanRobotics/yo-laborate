import configparser
import json
import sys
from pathlib import Path
from typing import Iterable, Dict, Optional, List, Tuple

CONFIG_INI = "config.ini"

GIT = "GIT"
EXE_PATH = "EXE_PATH"
BRANCH_NAME = "BRANCH_NAME"
REMOTE_NAME = "REMOTE_NAME"

DATASET = "DATASET"
ROOT = "ROOT"
CLASSES_JSON = "CLASSES_JSON"
CLASSES_JSON_FILENAME = "classes.json"

YOLO = "YOLO"
YOLO_ROOT = "YOLO_ROOT"
HYP_PATH = "HYP_PATH"
WEIGHTS_PATH = "WEIGHTS_PATH"
CFG_PATH = "CFG_PATH"
PYTHON_EXE = "PYTHON_EXE"

YOLO_ANNOTATIONS_FOLDER_NAME = "YOLO_darknet"
LABELS_FOLDER_NAME = "labels"
PASCAL_VOC_FOLDER_NAME = "PASCAL_VOC"

ORANGE = "orange"
GREEN = "green"
RED = "red"
PURPLE = "purple"

RESULTS_FOLDER = ".results"
PERFORMANCE_FOLDER = ".performance"


def get_all_jpg_recursive(img_root: Optional[Path]) -> Iterable[Path]:
    if img_root.exists():
        items = img_root.rglob("*.jpg")
    else:
        print(f"WARNING. root_dir does not exist: {img_root}")
        items = []
    for item in items:
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
    config_path = base_dir / CONFIG_INI
    if not config_path.exists():
        raise RuntimeError(f"{str(config_path)} does not exist.")
    config.read(str(config_path))
    python_path = config.get(YOLO, PYTHON_EXE)
    yolo_root = config.get(YOLO, YOLO_ROOT)
    cfg_path = config.get(YOLO, CFG_PATH)
    weights_path = config.get(YOLO, WEIGHTS_PATH)
    hyp_path = config.get(YOLO, HYP_PATH)
    dataset_root = config.get(DATASET, ROOT)
    classes_json_path = config.get(DATASET, CLASSES_JSON)
    if (
        classes_json_path is None
        or classes_json_path == ""
        or classes_json_path == "./"
    ):
        classes_json_path = base_dir / CLASSES_JSON_FILENAME
    return (
        python_path,
        yolo_root,
        cfg_path,
        weights_path,
        hyp_path,
        dataset_root,
        classes_json_path,
    )


def get_yolo_detect_paths(base_dir: Path) -> Tuple[Path, Path]:
    config = configparser.ConfigParser()
    config_path = base_dir / CONFIG_INI
    if not config_path.exists():
        raise RuntimeError(f"{str(config_path)} does not exist.")
    config.read(str(config_path))
    python_path = config.get(YOLO, PYTHON_EXE)
    yolo_root = config.get(YOLO, YOLO_ROOT)
    return Path(python_path), Path(yolo_root)


def get_classes_list(base_dir: Path) -> List[str]:
    """
    Returns a list of class labels based on the "label" field in
    the classes.json file found in the base_dir.

    """
    config = configparser.ConfigParser()
    config_path = base_dir / CONFIG_INI
    if not config_path.exists():
        raise RuntimeError(f"{str(config_path)} does not exist.")
    config.read(str(config_path))
    classes_json_path = config.get(DATASET, CLASSES_JSON)
    if (
        classes_json_path is None
        or classes_json_path == ""
        or classes_json_path == "./"
    ):
        classes_json_path = base_dir / CLASSES_JSON_FILENAME
    else:
        classes_json_path = Path(classes_json_path).resolve()
    if not classes_json_path.exists():
        raise RuntimeError(
            f"CLASSES_JSON path does not exist at {str(classes_json_path)}"
        )
    classes_id_to_label_map = get_id_to_label_map(classes_json_path=classes_json_path)
    class_labels_list = list(classes_id_to_label_map.values())
    return class_labels_list


def get_version_control_config(base_dir: Path = Path(__file__).parents[1]):
    config = configparser.ConfigParser()
    config_path = base_dir / CONFIG_INI
    if not config_path.exists():
        raise RuntimeError(f"{str(config_path)} does not exist.")
    config.read(str(config_path))
    git_exe_path = str(Path(config.get(GIT, EXE_PATH)).resolve())
    remote_name = config.get(GIT, REMOTE_NAME)
    branch_name = config.get(GIT, BRANCH_NAME)
    return git_exe_path, remote_name, branch_name


def inferred_base_dir() -> Path:
    """
    Infers the base_dir based on either the calling script or
    the current working directory, then checks the config.ini
    to check for rerouting to another root.

    Keep in mind that a config.ini file could define DATASET:ROOT
    as a directory other than itself for testing purposes.

    """
    cwd = Path().cwd()
    caller = Path(sys.argv[0])

    if caller.name == "label_folder" and Path(caller.parents[2] / CONFIG_INI).exists():
        base_dir = caller.parents[2]
    elif (cwd / CONFIG_INI).exists() and (cwd / CLASSES_JSON_FILENAME).exists():
        base_dir = cwd
    elif (cwd.parent / CONFIG_INI).exists() and (
        cwd.parent / CLASSES_JSON_FILENAME
    ).exists():
        base_dir = cwd.parent
    elif (cwd.parents[1] / CONFIG_INI).exists() and (
        cwd.parents[1] / CLASSES_JSON_FILENAME
    ).exists():
        base_dir = cwd.parents[1]
    elif (cwd.parents[2] / CONFIG_INI).exists() and (
        cwd.parents[2] / CLASSES_JSON_FILENAME
    ).exists():
        base_dir = cwd.parents[2]
    elif (cwd.parents[3] / CONFIG_INI).exists() and (
        cwd.parents[3] / CLASSES_JSON_FILENAME
    ).exists():
        base_dir = cwd.parents[3]
    else:
        raise RuntimeError("Could not infer BASE_DIR.")

    """ Now check for re-routing to another directory. """
    config = configparser.ConfigParser()
    config_path = base_dir / CONFIG_INI
    if not config_path.exists():
        raise RuntimeError(f"{str(config_path)} does not exist.")
    config.read(str(config_path))
    root_dir = config.get(DATASET, ROOT)

    """ Return statements go below here """
    if root_dir is not None and root_dir != "./":
        tentative_dir = Path(root_dir).resolve()
    else:
        return base_dir
    if not tentative_dir.exists():
        raise RuntimeError(f"Path does not exist: {str(tentative_dir)}")
    if str(tentative_dir) != str(root_dir):
        print(f"Testing mode. Rerouting to dataset at: {str(tentative_dir)}")
    if (tentative_dir / "classes.json").exists():
        return tentative_dir
    else:
        print(
            "Path exists but looks suspect because "
            "it does not contain a file classes.json. "
            f"Path: {str(tentative_dir)}"
        )
        return tentative_dir


def save_output_to_text_file(
    content: str,
    base_dir: Path,
    file_name: str,
    commit: bool = False,
):
    if commit:
        folder_name = PERFORMANCE_FOLDER
    else:
        folder_name = RESULTS_FOLDER
    output_path = base_dir / folder_name / file_name
    output_path.parent.mkdir(exist_ok=True)
    with open(str(output_path), "w") as file_out:
        file_out.write(content)
