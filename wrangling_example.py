from pathlib import Path

from yo_ratchet.dataset_versioning.version import get_dataset_label_from_version

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
