from pathlib import Path

from common import get_config_items, get_id_to_label_map
from fiftyone_integration import init_fifty_one_dataset, delete_fiftyone_dataset, find_errors
from wrangle import run_detections, prepare_dataset_and_train, reverse_train
from wrangling_example import SUBSETS_INCLUDED, EVERY_NTH_TO_VAL, KEEP_CLASS_IDS, SKIP_CLASS_IDS, DATASET_LABEL, \
    REVERSE_TRAIN_VAL, CONF, GROUPINGS
from yo_valuate import binary_and_group_classification_performance


_, yolo_root, _, _, _, dataset_root, classes_json_path = get_config_items(
    base_dir=Path(__file__).parent
)
YOLO_ROOT = Path(yolo_root)
DATASET_ROOT = Path(dataset_root)
CLASSES_JSON_PATH = Path(classes_json_path)
CLASSES_MAP = get_id_to_label_map(CLASSES_JSON_PATH)
DST_ROOT = Path(YOLO_ROOT) / f"datasets/{DATASET_LABEL}"
CONFIDENCE = int(CONF * 100)

if REVERSE_TRAIN_VAL:
    TEST_SET_LABEL = "train"
    MODEL_LABEL = f"{DATASET_LABEL}_reverse"

else:
    TEST_SET_LABEL = "val"
    MODEL_LABEL = DATASET_LABEL

PROCESSED_ROOT = DST_ROOT / TEST_SET_LABEL
TEST_IMAGES_ROOT = DST_ROOT / TEST_SET_LABEL / "images"
GROUND_TRUTHS_PATH = DST_ROOT / TEST_SET_LABEL / "labels"

TEST_DATASET_PART_LABEL = f"{DATASET_LABEL}_{TEST_SET_LABEL}"
INFERENCE_RUN_NAME = f"{TEST_DATASET_PART_LABEL}__{MODEL_LABEL}_conf{CONFIDENCE}pcnt"
INFERENCES_PATH = YOLO_ROOT / f"runs/detect/{INFERENCE_RUN_NAME}/labels"


def run_prepare_dataset_and_train():
    print(__file__, __name__)
    prepare_dataset_and_train(
        classes_map=CLASSES_MAP,
        subsets_included=SUBSETS_INCLUDED,
        dst_root=DST_ROOT,
        every_n_th=EVERY_NTH_TO_VAL,
        keep_class_ids=KEEP_CLASS_IDS,
        skip_class_ids=SKIP_CLASS_IDS,
        base_dir=Path(__file__).parent,
        run_training=False,
    )
    run_detections(
        images_path=TEST_IMAGES_ROOT,
        dataset_version=TEST_DATASET_PART_LABEL,
        model_path=Path(f"{YOLO_ROOT}/runs/train/{MODEL_LABEL}/weights/best.pt"),
        model_version=MODEL_LABEL,
        base_dir=Path(__file__).parent,
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
    with open(f"{MODEL_LABEL}.txt", "w") as file_out:
        file_out.write(table_str)
    delete_fiftyone_dataset(dataset_label=DATASET_LABEL)
    init_fifty_one_dataset(
        dataset_label=DATASET_LABEL,
        classes_map=CLASSES_MAP,
        inferences_root=INFERENCES_PATH,
        processed_root=PROCESSED_ROOT,
        dataset_root=dataset_root,
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


def test_init_fiftyone_ds():
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


def test_find_errors(tag="mistakenness"):
    find_errors(
        dataset_label=DATASET_LABEL,
        class_names=list(CLASSES_MAP.values()),
        tag=tag,
        conf_thresh=0.1,
        limit=32,
        processed=True,
        reverse=True,
        label_filter="WS",
    )
