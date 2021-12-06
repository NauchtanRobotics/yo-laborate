from pathlib import Path

from wrangle import prepare_dataset_and_train

SUBSETS_INCLUDED = [
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Caboone_10pcnt_AP_LO_LG_WS"),
        313,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Caboone_40pcnt_D10_D20_D40_EB"),
        415,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/CentralCoast_10pcnt_L0_LG_WS"),
        475,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/CentralCoast_25pcnt_AP_D10_D20"),
        626,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/CentralCoast_35pct_EB"),
        87,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Gladstone_2020_sample_1"),
        621,
    ),
    (
        Path("/home/david/RACAS/sealed_roads_dataset/Train_Gladstone_2020_sample_2"),
        503,
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
DST_ROOT = Path("/home/david/addn_repos/yolov5/datasets/vvvtest")
CLASSES_LIST_PATH = Path(
    "/home/david/addn_repos/OpenLabeling/open_labeling/class_list.txt"
)


def test_prepare_dataset_and_train():
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
