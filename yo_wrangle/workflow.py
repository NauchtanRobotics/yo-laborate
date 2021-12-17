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
MODEL_LABEL = ""
PROCESSED_ROOT = Path()
TEST_IMAGES_ROOT = Path()
GROUND_TRUTHS_PATH = Path()
TEST_DATASET_PART_LABEL = ""
INFERENCE_RUN_NAME = ""
INFERENCES_PATH = Path()
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
    global DST_ROOT, CONFIDENCE, TEST_SET_LABEL, MODEL_LABEL
    global PROCESSED_ROOT, TEST_IMAGES_ROOT, GROUND_TRUTHS_PATH
    global TEST_DATASET_PART_LABEL, INFERENCE_RUN_NAME, INFERENCES_PATH
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

    if workbook_ptr.REVERSE_TRAIN_VAL:
        TEST_SET_LABEL = "train"
        MODEL_LABEL = f"{workbook_ptr.DATASET_LABEL}_reverse"

    else:
        TEST_SET_LABEL = "val"
        MODEL_LABEL = workbook_ptr.DATASET_LABEL

    PROCESSED_ROOT = DST_ROOT / TEST_SET_LABEL
    TEST_IMAGES_ROOT = DST_ROOT / TEST_SET_LABEL / "images"
    GROUND_TRUTHS_PATH = DST_ROOT / TEST_SET_LABEL / "labels"

    TEST_DATASET_PART_LABEL = f"{workbook_ptr.DATASET_LABEL}_{TEST_SET_LABEL}"
    INFERENCE_RUN_NAME = (
        f"{TEST_DATASET_PART_LABEL}__{MODEL_LABEL}_conf{CONFIDENCE}pcnt"
    )
    INFERENCES_PATH = YOLO_ROOT / f"runs/detect/{INFERENCE_RUN_NAME}/labels"


def run_prepare_dataset_and_train(run_training=True):
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
    run_detections(
        images_path=TEST_IMAGES_ROOT,
        dataset_version=TEST_DATASET_PART_LABEL,
        model_path=Path(f"{YOLO_ROOT}/runs/train/{MODEL_LABEL}/weights/best.pt"),
        model_version=MODEL_LABEL,
        base_dir=BASE_DIR,
        conf_thres=CONF,
        device=0,
    )
    table_str = binary_and_group_classification_performance(
        images_root=TEST_IMAGES_ROOT,
        root_ground_truths=GROUND_TRUTHS_PATH,
        root_inferred_bounding_boxes=INFERENCES_PATH,
        classes_map=CLASSES_MAP,
        print_first_n=24,
        groupings=GROUPINGS,
    )
    output_filename = (
        f"{MODEL_LABEL}_classification_performance_conf{CONFIDENCE}pcnt.txt"
    )
    with open(output_filename, "w") as file_out:
        file_out.write(table_str)
    delete_fiftyone_dataset(dataset_label=DATASET_LABEL)
    init_fifty_one_dataset(
        dataset_label=DATASET_LABEL,
        classes_map=CLASSES_MAP,
        inferences_root=INFERENCES_PATH,
        processed_root=PROCESSED_ROOT,
        dataset_root=DATASET_ROOT,
        images_root=None,  # Use dataset_root approach
        ground_truths_root=None,  # Use dataset_root approach
    )


def run_reverse_train():
    reverse_train(
        classes_map=CLASSES_MAP,
        dst_root=DST_ROOT,
        base_dir=Path(__file__).parent,
    )
    run_detections(
        images_path=TEST_IMAGES_ROOT,
        dataset_version=TEST_DATASET_PART_LABEL,
        model_path=Path(
            f"/home/david/addn_repos/yolov5/runs/train/{MODEL_LABEL}/weights/best.pt"
        ),
        model_version=MODEL_LABEL,
        base_dir=Path(__file__).parent,
        conf_thres=CONF,
        device=1,
    )
    table_str = binary_and_group_classification_performance(
        images_root=TEST_IMAGES_ROOT,
        root_ground_truths=GROUND_TRUTHS_PATH,
        root_inferred_bounding_boxes=INFERENCES_PATH,
        classes_map=CLASSES_MAP,
        print_first_n=24,
        groupings=GROUPINGS,
    )
    with open(f"{MODEL_LABEL}.txt", "w") as file_out:
        file_out.write(table_str)
    test_init_fiftyone_ds()


def test_init_fiftyone_ds(candidate_subset: Path = None):
    delete_fiftyone_dataset(dataset_label=DATASET_LABEL)
    init_fifty_one_dataset(
        dataset_label=DATASET_LABEL,
        classes_map=CLASSES_MAP,
        inferences_root=INFERENCES_PATH,
        processed_root=PROCESSED_ROOT,
        dataset_root=DATASET_ROOT,
        images_root=None,  # Use dataset_root approach
        ground_truths_root=None,  # Use dataset_root approach
        candidate_subset=candidate_subset,
    )


def test_find_errors(tag="mistakenness"):
    find_errors(
        dataset_label=DATASET_LABEL,
        class_names=list(CLASSES_MAP.values()),
        tag=tag,
        limit=32,
        processed=True,
        reverse=True,
        label_filter="WS",
    )
