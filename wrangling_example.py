from pathlib import Path

from common import get_id_to_label_map, get_config_items
from fiftyone_integration import init_fifty_one_dataset
from wrangle import prepare_dataset_and_train, run_detections, reverse_train

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
        580,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Weddin_2019_samples"),
        None,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_WDRC_2021_sample_1"),
        504,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Whitsunday_2018_samples"),
        None,
    ),
    # (
    #     Path(
    #         "/home/david/RACAS/boosted/600_x_600/unmasked/Train_Charters_Towers_2021_subsample"
    #     ),
    #     None,
    # ),
]

KEEP_CLASS_IDS = None  # None actually means keep all classes
SKIP_CLASS_IDS = [10, 13, 14, 15, 22]  # AP, RK, SD, S, Sh
EVERY_NTH_TO_VAL = 1  # for the validation subset
DATASET_LABEL = "v8a"


# DERIVED CONSTANTS
_, yolo_root, _, _, _, dataset_root, classes_list = get_config_items(base_dir=Path(__file__).parent)
YOLO_ROOT = Path(yolo_root)
DATASET_ROOT = Path(dataset_root)
CLASSES_LIST_PATH = Path(classes_list)
DST_ROOT = Path(YOLO_ROOT) / f"datasets/{DATASET_LABEL}"


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


def test_run_detections():
    print(__file__, __name__)
    model_name = DST_ROOT.name
    run_detections(
        images_path=(DST_ROOT / "val" / "images"),
        dataset_version=f"{DST_ROOT.name}_val",
        model_path=Path(f"{YOLO_ROOT}/runs/train/{model_name}/weights/best.pt"),
        model_version=DST_ROOT.name,
        base_dir=Path(__file__).parent,
        conf_thres=0.1,
        device=0,
    )


def test_init_fiftyone_ds():
    """Creates a fiftyone dataset. Currently, this initialiser
    relies on hard coded local variables.

    """
    import fiftyone as fo

    label_mapping = get_id_to_label_map(CLASSES_LIST_PATH)

    images_root = None
    ground_truths_root = None
    inference_run_name = f"{DATASET_LABEL}_val__{DATASET_LABEL}_conf10pcnt"
    inferences_root = (
        YOLO_ROOT / f"runs/detect/{inference_run_name}/labels"
    )

    dataset_label = DST_ROOT.name
    if dataset_label in fo.list_datasets():
        fo.delete_dataset(name=dataset_label)
    else:
        pass
    processed_root = DST_ROOT / "val"
    init_fifty_one_dataset(
        dataset_label=dataset_label,
        label_mapping=label_mapping,
        inferences_root=inferences_root,
        processed_root=processed_root,
        dataset_root=DATASET_ROOT,
        images_root=images_root,
        ground_truths_root=ground_truths_root,
    )


def run_reverse_train():
    reverse_train(
        class_list_path=CLASSES_LIST_PATH,
        dst_root=DST_ROOT,
        base_dir=Path(__file__).parent,
    )


def test_run_reverse_detections():
    print(__file__, __name__)
    model_name = f"{DST_ROOT.name}_reverse"
    run_detections(
        images_path=(DST_ROOT / "train" / "images"),
        dataset_version=f"{DST_ROOT.name}_train",
        model_path=Path(f"/home/david/addn_repos/yolov5/runs/train/{model_name}/weights/best.pt"),
        model_version=model_name,
        base_dir=Path(__file__).parent,
        conf_thres=0.1,
        device=1,
    )
