from pathlib import Path

from dataset_versioning.version import get_dataset_label_from_version

SUBSETS_INCLUDED = [
    (Path("Caboone_10pcnt_AP_LO_LG_WS"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("Caboone_40pcnt_D10_D20_D40_EB"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("CentralCoast_10pcnt_L0_LG_WS"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("CentralCoast_25pcnt_AP_D10_D20"), None),
    (Path("CentralCoast_35pct_EB"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("Train_Gladstone_2020_sample_1"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("Train_Gladstone_2020_sample_2"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("Train_Gladstone_2020_sample_3"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("Train_Huon_2021_sample_1"), None),
    (Path("Train_Isaac_2021_sample_1"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("Train_Isaac_2021_sample_2"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("Train_Maranoa_2020_sample_1"), None),
    (Path("Train_Maranoa_2020_sample_2"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("Train_Maranoa_2020_sample_3"), None),
    (Path("Train_Moira_2020_sample_1"), None),
    (Path("YOLO_Moira_2020_sample_val"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("YOLO_Hobart_2021_sample"), None),
    (Path("YOLO_Isaac_2021_pothole_sample_val"), None),  # Done
    (Path("Train_Weddin_2019_samples"), None),  # Done
    (Path("Train_WDRC_2021_sample_1"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("Train_Whitsunday_2018_samples"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("Train_Charters_Towers_2021_subsample"), None),  # Done
    (Path("Train_Charters_Towers_2021_WS"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("Isaac_stripping"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("WDRC_Stripping"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("Central_Coast_Stripping"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("CT_EB_D40_Cracking_hard_pos"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("CT_Stripping"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("CT_D40_SD"), None),  # Done Stp, Stp2, PP, PPF, Scf, Stp1
    (Path("Scenic_Rim_2021_mined_1"), None),  # Done/New
    (Path("Scenic_Rim_2022_mined_1"), None)  # Done/New
    # DON'T MAKE UPDATES HERE. DO IT IN THE RESPECTIVE DATASET REPO (e.g. sealed_roads_dataset).
]

KEEP_CLASS_IDS = None  # None actually means keep all classes
SKIP_CLASS_IDS = [15, 22]  # Signs, Shoving
EVERY_NTH_TO_VAL = 1  # for the validation subset
DATASET_LABEL = get_dataset_label_from_version(Path(__file__).parent)
GROUPINGS = {
    "Risk Defects": [3, 4],
    "Potholes Big/Small": [3, 18],
    "Cracking": [0, 1, 2, 11, 14, 16],
    "Stripping": [12, 17, 18, 19, 20, 24, 25],
    # DON'T MAKE UPDATES HERE.
}
RECODE_MAP = {24: 12}
CONF = 0.1
