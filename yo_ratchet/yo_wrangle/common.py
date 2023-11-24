import configparser
import json
import sys
from pathlib import Path
from typing import Iterable, Dict, Optional, List, Tuple, Union

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

ENSEMBLE_MODEL_ROOT_TOML = "ENSEMBLE_MODEL_ROOT"
ENSEMBLE_MAX_COUNT_TOML = "ENSEMBLE_MAX_COUNT"
MODEL_VERSION_TOML = "MODEL_VERSION"

YOLO_ANNOTATIONS_FOLDER_NAME = "YOLO_darknet"
LABELS_FOLDER_NAME = "labels"
PASCAL_VOC_FOLDER_NAME = "PASCAL_VOC"

ORANGE = "orange"
GREEN = "green"
RED = "red"
PURPLE = "purple"

RESULTS_FOLDER = ".results"
PERFORMANCE_FOLDER = ".performance"


def get_subsets_included(base_dir: Path) -> List[Path]:
    included_subsets = []
    base_dir = base_dir.absolute()
    excluded_subsets_file = base_dir / "subsets_excluded.txt"
    if not excluded_subsets_file.exists():
        excluded_subsets_file = base_dir / "excluded_subsets.txt"

    if excluded_subsets_file.exists():
        with open(excluded_subsets_file, "r") as excluded:
            excluded_files = excluded.read().splitlines()
    else:
        excluded_files = []

    for item in sorted(base_dir.iterdir()):
        first_char = item.name[0]
        if first_char != "." and first_char != "_" and item.is_dir() and item.name not in excluded_files:
            included_subsets.append(item)
    return included_subsets


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


def get_label_to_id_map(base_dir: Path):
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
        id_to_label_map = get_id_to_label_map(classes_json_path)
        label_to_id = {
            class_label: class_id for class_id, class_label in id_to_label_map.items()
        }
        return label_to_id


def get_config_items(base_dir: Path) -> Tuple[str, str, str, str, str, str, Union[str, Path]]:
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


def get_implicit_model_paths(base_dir: Path, dataset_identifier: str) -> List[Path]:
    config = configparser.ConfigParser()
    config_path = base_dir / CONFIG_INI
    if not config_path.exists():
        raise RuntimeError(f"{str(config_path)} does not exist.")
    config.read(str(config_path))

    model_version: Optional[str] = None
    model_folders: Optional[List[Path]] = None
    ensemble_model_root: Optional[Path] = None
    try:
        ensemble_model_dir = config.get(dataset_identifier, ENSEMBLE_MODEL_ROOT_TOML)
        ensemble_model_dir = ensemble_model_dir.replace("~", str(Path().home()))
        ensemble_model_root = Path(ensemble_model_dir).resolve()

        model_version = ensemble_model_root.name
        model_folders = [model_folder for model_folder in ensemble_model_root.iterdir() if model_folder.is_dir()]
        if len(model_folders) == 0:
            raise RuntimeError("No sub-folders found in ensemble root dir: \n" + str(ensemble_model_root))
    except Exception as ex:
        pass

    if ensemble_model_root is not None and not ensemble_model_root.exists():
        raise RuntimeError("Path does not exist:\n" + str(ensemble_model_root))

    if model_folders is None:
        model_version = config.get(dataset_identifier, MODEL_VERSION_TOML)
        _, yolo_root = get_yolo_detect_paths(base_dir)
        yolo_root = yolo_root.resolve()
        models_root_path = yolo_root / "runs" / "train"  # assumed by past convention
        if not models_root_path.exists():
            raise RuntimeError("Path not found:\n" + str(models_root_path))

        model_folders = [folder for folder in models_root_path.glob(model_version + ".*")]
        if len(model_folders) == 0:
            model_folder = models_root_path / model_version
            if model_folder.exists():
                model_folders = [model_folder]
            else:
                raise RuntimeError("Could not find model folder: \n" + str(model_folder))
        else:
            pass  # We've found some promising looking sub-folders

    model_paths = [
        model_folder / "weights" / "best.pt" for model_folder in model_folders
        if (model_folder / "weights" / "best.pt").exists()
    ]
    if model_version is not None and len(model_paths) == 0:
        raise RuntimeError("No models found for model version " + model_version + "\n" +
                           "E.g. " + str(model_folders[0]) + "/weights/best.pt\n" +
                           "n.b. sub-folder structure /weights/best.pt is mandatory.")
    max_ensemble_count: Optional[int] = None
    try:
        max_ensemble_count = int(config.get(dataset_identifier, ENSEMBLE_MAX_COUNT_TOML))
    except:
        pass

    if max_ensemble_count and len(model_paths) > max_ensemble_count:
        model_paths = model_paths[:max_ensemble_count]
    else:
        pass
    return model_paths


def get_classes_json_path(base_dir: Path) -> Optional[Path]:
    """
    Returns a Path to the classes.json file found in the base_dir.

    """
    config = configparser.ConfigParser()
    config_path = base_dir / CONFIG_INI
    if not config_path.exists():
        raise RuntimeError(f"{str(config_path)} does not exist.")
    config.read(str(config_path))
    try:
        classes_json_path = config.get(DATASET, CLASSES_JSON)
    except:
        return None
    if (
        classes_json_path is None
        or classes_json_path == ""
        or classes_json_path == "./"
    ):
        classes_json_path = base_dir / CLASSES_JSON_FILENAME
    else:
        classes_json_path = Path(classes_json_path).resolve()
    return classes_json_path


def get_classes_list(base_dir: Path) -> List[str]:
    """
    Returns a list of class labels based on the "label" field in
    the classes.json file found in the base_dir.

    """
    classes_json_path = get_classes_json_path(base_dir=base_dir)
    if classes_json_path is None:
        raise RuntimeError(str(base_dir) + "/classes.json could not be found.")
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
        print("Could not infer BASE_DIR.")
        base_dir = None
        return base_dir

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
