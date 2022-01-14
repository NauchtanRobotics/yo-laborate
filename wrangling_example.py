from pathlib import Path

from dataset_versioning.version import get_dataset_label_from_version

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
            "/home/david/RACAS/sealed_roads_dataset/Train_Charters_Towers_2021_subsample"
        ),
        None,
    ),
]

KEEP_CLASS_IDS = None  # None actually means keep all classes
SKIP_CLASS_IDS = [15, 22]  # Signs, Shoving
EVERY_NTH_TO_VAL = 1  # for the validation subset
DATASET_LABEL = get_dataset_label_from_version(Path(__file__).parent)
GROUPINGS = {"Risk Defects": [3, 4, 14], "Cracking": [0, 1, 2, 11, 16]}
CONF = 0.1
