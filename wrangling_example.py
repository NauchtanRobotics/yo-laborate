from pathlib import Path

from common import get_id_to_label_map, get_config_items
from fiftyone_integration import init_fifty_one_dataset, find_errors
from wrangle import prepare_dataset_and_train, run_detections, reverse_train
from yo_valuate import binary_and_group_classification_performance

SUBSETS_INCLUDED = [
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Caboone_10pcnt_AP_LO_LG_WS"),
        None,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Caboone_40pcnt_D10_D20_D40_EB"),
        None,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/CentralCoast_10pcnt_L0_LG_WS"),
        None,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/CentralCoast_25pcnt_AP_D10_D20"),
        None,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/CentralCoast_35pct_EB"),
        None,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Gladstone_2020_sample_1"),
        None,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Gladstone_2020_sample_2"),
        None,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Gladstone_2020_sample_3"),
        163,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Huon_2021_sample_1"),
        398,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Isaac_2021_sample_1"),
        579,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Isaac_2021_sample_2"),
        525,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Maranoa_2020_sample_1"),
        175,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Maranoa_2020_sample_2"),
        502,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Maranoa_2020_sample_3"),
        175,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Moira_2020_sample_1"),
        670,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/YOLO_Moira_2020_sample_val"),
        670,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/YOLO_Hobart_2021_sample"),
        492,
    ),
    (
        Path(
            "/home/david/RACAS/sealed_roads_dataset/YOLO_Isaac_2021_pothole_sample_val"
        ),
        None,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Weddin_2019_samples"),
        None,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Whitsunday_2018_samples"),
        None,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_WDRC_2021_sample_1"),
        None,
    ),
    (
        Path(
            "/home/david/RACAS/boosted/600_x_600/unmasked/Train_Charters_Towers_2021_subsample"
        ),
        None,
    ),
]

KEEP_CLASS_IDS = None  # None actually means keep all classes
SKIP_CLASS_IDS = [15, 22]  # Signs, Shoving
EVERY_NTH_TO_VAL = 1  # for the validation subset
DATASET_LABEL = "v8d"
REVERSE_TRAIN_VAL = False
GROUPINGS = {"Risk Defects": [3, 4, 14], "Cracking": [0, 1, 2, 11, 16]}
CONF = 0.1

# DERIVED CONSTANTS
_, yolo_root, _, _, _, dataset_root, classes_list = get_config_items(
    base_dir=Path(__file__).parent
)
YOLO_ROOT = Path(yolo_root)
DATASET_ROOT = Path(dataset_root)
CLASSES_LIST_PATH = Path(classes_list)
CLASSES_MAPPING = get_id_to_label_map(CLASSES_LIST_PATH)
DST_ROOT = Path(YOLO_ROOT) / f"datasets/{DATASET_LABEL}"
CONFIDENCE = int(CONF*100)

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
        class_list_path=CLASSES_LIST_PATH,
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
        class_names_path=CLASSES_LIST_PATH,
        print_first_n=24,
        groupings=GROUPINGS,
    )
    with open(f"{MODEL_LABEL}.txt", "w") as file_out:
        file_out.write(table_str)
    test_init_fiftyone_ds()
   
   
def test_init_fiftyone_ds():
    """Creates a fiftyone dataset. Currently, this initialiser
    relies on hard coded local variables.

    """
    import fiftyone as fo

    if DATASET_LABEL in fo.list_datasets():
        fo.delete_dataset(name=DATASET_LABEL)
    else:
        pass

    init_fifty_one_dataset(
        dataset_label=DATASET_LABEL,
        label_mapping=CLASSES_MAPPING,
        inferences_root=INFERENCES_PATH,
        processed_root=PROCESSED_ROOT,
        dataset_root=DATASET_ROOT,
        images_root=None,  # Use dataset_root approach
        ground_truths_root=None,  # Use dataset_root approach
    )


def run_reverse_train():
    reverse_train(
        class_list_path=CLASSES_LIST_PATH,
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
        class_names_path=CLASSES_LIST_PATH,
        print_first_n=24,
        groupings=GROUPINGS,
    )
    with open(f"{MODEL_LABEL}.txt", "w") as file_out:
        file_out.write(table_str)
    test_init_fiftyone_ds()


def test_find_errors(tag="mistakenness"):
    find_errors(
        dataset_label=DATASET_LABEL,
        tag=tag,
        conf_thresh=0.1,
        limit=32,
        processed=True,
        reverse=True,
        label_filter="WS",
    )
