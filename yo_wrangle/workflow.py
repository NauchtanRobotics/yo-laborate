from pathlib import Path

from yo_wrangle.common import get_config_items, get_id_to_label_map
from yo_wrangle.fiftyone_integration import (
    init_fifty_one_dataset,
    delete_fiftyone_dataset,
    find_errors,
)
from yo_wrangle.wrangle import run_detections, prepare_dataset_and_train, reverse_train
from yo_wrangle.yo_valuate import binary_and_group_classification_performance

YOLO_ROOT = Path()
DATASET_ROOT = Path()
CLASSES_JSON_PATH = Path()
CLASSES_MAP = {}
DST_ROOT = Path()
CONFIDENCE = 0.1
TEST_SET_LABEL = ""
BASE_DIR = Path()
CONF = 0.05
GROUPINGS = {}

SUBSETS_INCLUDED = []
EVERY_NTH_TO_VAL = 0
KEEP_CLASS_IDS = []
SKIP_CLASS_IDS = []
DATASET_LABEL = ""


def set_globals(base_dir: Path, workbook_ptr):
    global YOLO_ROOT, DATASET_ROOT, CLASSES_JSON_PATH, CLASSES_MAP
    global DST_ROOT, CONFIDENCE
    global BASE_DIR, CONF, GROUPINGS, SUBSETS_INCLUDED, EVERY_NTH_TO_VAL
    global KEEP_CLASS_IDS, SKIP_CLASS_IDS, DATASET_LABEL

    CONF = workbook_ptr.CONF
    SUBSETS_INCLUDED = workbook_ptr.SUBSETS_INCLUDED
    EVERY_NTH_TO_VAL = workbook_ptr.EVERY_NTH_TO_VAL
    KEEP_CLASS_IDS = workbook_ptr.KEEP_CLASS_IDS
    SKIP_CLASS_IDS = workbook_ptr.SKIP_CLASS_IDS
    DATASET_LABEL = workbook_ptr.DATASET_LABEL

    GROUPINGS = workbook_ptr.GROUPINGS
    BASE_DIR = base_dir
    _, yolo_root, _, _, _, dataset_root, classes_json_path = get_config_items(
        base_dir=base_dir
    )
    YOLO_ROOT = Path(yolo_root)
    DATASET_ROOT = Path(dataset_root)
    CLASSES_JSON_PATH = Path(classes_json_path)
    CLASSES_MAP = get_id_to_label_map(CLASSES_JSON_PATH)
    DST_ROOT = Path(YOLO_ROOT) / f"datasets/{workbook_ptr.DATASET_LABEL}"
    CONFIDENCE = int(workbook_ptr.CONF * 100)


def get_labels_and_paths_tuple(dataset_label: str, reverse_it: bool = False):
    if reverse_it:
        test_set_str = "train"
        model_label = f"{dataset_label}_reverse"

    else:
        test_set_str = "val"
        model_label = dataset_label
    test_set_part_label = f"{dataset_label}_{test_set_str}"
    ground_truth_path = DST_ROOT / test_set_str / "labels"
    run_name = f"{test_set_part_label}__{model_label}_conf{CONFIDENCE}pcnt"
    inferences_path = YOLO_ROOT / f"runs/detect/{run_name}/labels"
    return model_label, test_set_part_label, ground_truth_path, inferences_path


def run_prepare_dataset_and_train(run_training=True, init_fiftyone=True):
    prepare_dataset_and_train(
        classes_map=CLASSES_MAP,
        subsets_included=SUBSETS_INCLUDED,
        dst_root=DST_ROOT,
        every_n_th=EVERY_NTH_TO_VAL,
        keep_class_ids=KEEP_CLASS_IDS,
        skip_class_ids=SKIP_CLASS_IDS,
        base_dir=BASE_DIR,
        run_training=run_training,
    )
    detect_images_root = DST_ROOT / "val" / "images"
    (
        model_label,
        test_set_part_label,
        ground_truth_path,
        inferences_path,
    ) = get_labels_and_paths_tuple(dataset_label=DATASET_LABEL, reverse_it=False)
    run_detections(
        images_path=detect_images_root,
        dataset_version=test_set_part_label,
        model_path=Path(f"{YOLO_ROOT}/runs/train/{model_label}/weights/best.pt"),
        model_version=model_label,
        base_dir=BASE_DIR,
        conf_thres=CONF,
        device=0,
    )
    table_str = binary_and_group_classification_performance(
        images_root=detect_images_root,
        root_ground_truths=ground_truth_path,
        root_inferred_bounding_boxes=inferences_path,
        classes_map=CLASSES_MAP,
        print_first_n=24,
        groupings=GROUPINGS,
    )
    output_filename = f"{model_label}_forward_performance_conf{CONFIDENCE}pcnt.txt"
    with open(output_filename, "w") as file_out:
        file_out.write(table_str)
    delete_fiftyone_dataset(dataset_label=DATASET_LABEL)
    if init_fiftyone:
        init_fifty_one_dataset(
            dataset_label=DATASET_LABEL,
            classes_map=CLASSES_MAP,
            train_inferences_root=None,
            val_inferences_root=inferences_path,
            dataset_root=DATASET_ROOT,
            images_root=None,  # Use dataset_root approach
            ground_truths_root=None,  # Use dataset_root approach
        )


def run_reverse_train(init_fiftyone: bool = True):
    reverse_train(
        classes_map=CLASSES_MAP,
        dst_root=DST_ROOT,
        base_dir=BASE_DIR,
    )
    detect_images_root = DST_ROOT / "train" / "images"
    (
        model_label,
        test_set_part_label,
        ground_truth_path,
        inferences_path,
    ) = get_labels_and_paths_tuple(dataset_label=DATASET_LABEL, reverse_it=True)
    run_detections(
        images_path=detect_images_root,
        dataset_version=test_set_part_label,
        model_path=Path(
            f"/home/david/addn_repos/yolov5/runs/train/{model_label}/weights/best.pt"
        ),
        model_version=model_label,
        base_dir=BASE_DIR,
        conf_thres=CONF,
        device=1,
    )
    table_str = binary_and_group_classification_performance(
        images_root=detect_images_root,
        root_ground_truths=ground_truth_path,
        root_inferred_bounding_boxes=inferences_path,
        classes_map=CLASSES_MAP,
        print_first_n=24,
        groupings=GROUPINGS,
    )
    output_filename = f"{model_label}_reverse_performance_conf{CONFIDENCE}pcnt.txt"
    with open(output_filename, "w") as file_out:
        file_out.write(table_str)
    delete_fiftyone_dataset(dataset_label=DATASET_LABEL)
    if init_fiftyone:
        init_fifty_one_dataset(
            dataset_label=DATASET_LABEL,
            classes_map=CLASSES_MAP,
            train_inferences_root=inferences_path,
            val_inferences_root=None,
            dataset_root=DATASET_ROOT,
            images_root=None,  # Use dataset_root approach
            ground_truths_root=None,  # Use dataset_root approach
        )


def run_full_training():
    """
    Runs forward and reverse training, where forward means train=train and val=val;
    and reverse means train=val, val=train.

    """
    # run_prepare_dataset_and_train(init_fiftyone=False)
    # run_reverse_train(init_fiftyone=False)
    delete_fiftyone_dataset(dataset_label=DATASET_LABEL)
    (_, _, _, val_inferences_root) = get_labels_and_paths_tuple(
        dataset_label=DATASET_LABEL, reverse_it=False
    )
    (_, _, _, train_inferences_root) = get_labels_and_paths_tuple(
        dataset_label=DATASET_LABEL, reverse_it=True
    )
    init_fifty_one_dataset(
        dataset_label=DATASET_LABEL,
        classes_map=CLASSES_MAP,
        train_inferences_root=train_inferences_root,
        val_inferences_root=val_inferences_root,
        dataset_root=DATASET_ROOT,
        images_root=None,  # Use dataset_root approach
        ground_truths_root=None,  # Use dataset_root approach
    )


def test_init_fiftyone_ds(candidate_subset: Path = None):
    delete_fiftyone_dataset(dataset_label=DATASET_LABEL)
    delete_fiftyone_dataset(dataset_label=DATASET_LABEL)
    (_, _, _, val_inferences_root) = get_labels_and_paths_tuple(
        dataset_label=DATASET_LABEL, reverse_it=False
    )
    init_fifty_one_dataset(
        dataset_label=DATASET_LABEL,
        classes_map=CLASSES_MAP,
        val_inferences_root=val_inferences_root,
        train_inferences_root=None,
        dataset_root=DATASET_ROOT,
        images_root=None,  # Use dataset_root approach
        ground_truths_root=None,  # Use dataset_root approach
        candidate_subset=candidate_subset,
    )


def run_find_errors(
    tag: str = "mistakenness",
    label_filter: str = "WS",
    limit: int = 64,
    dataset_label: str = None,
):
    if dataset_label is None:
        dataset_label = DATASET_LABEL
    else:
        pass
    find_errors(
        dataset_label=dataset_label,
        class_names=list(CLASSES_MAP.values()),
        tag=tag,
        limit=limit,
        processed=True,
        reverse=True,
        label_filter=label_filter,
    )
