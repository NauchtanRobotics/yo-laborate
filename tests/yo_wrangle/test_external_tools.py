import fiftyone as fo
from pathlib import Path

from yo_ratchet.yo_wrangle.common import get_id_to_label_map, get_config_items
from yo_ratchet.fiftyone_integration import init_fifty_one_dataset

TEST_DATA_ROOT = Path(__file__).parents[1] / "tests/test_data"
BASE_DIR = Path(__file__).parents[2]
_, _, _, _, _, _, CLASSES_JSON_PATH = get_config_items(base_dir=BASE_DIR)


def test_init_fifty_one_dataset():
    """
    TODO: - Add a class list to the test_data.
          - Commit test data on Windows machine.
    """
    dataset_label = "test_collation"
    label_mapping = get_id_to_label_map(classes_json_path=CLASSES_JSON_PATH)

    if dataset_label in fo.list_datasets():
        fo.delete_dataset(name=dataset_label)
    else:
        pass
    init_fifty_one_dataset(
        dataset_label=dataset_label,
        val_inferences_root=(
            TEST_DATA_ROOT
            / "runs/detect/Coll_7_train_Collation_7_scale40pcnt_10conf/labels"
        ),
        train_inferences_root=None,
        classes_map=label_mapping,
        images_root=(TEST_DATA_ROOT / "datasets/bbox_collation_7_split/train/images"),
        ground_truths_root=(
            TEST_DATA_ROOT / "datasets/bbox_collation_7_split/train/labels"
        ),
        candidate_subset=None,
    )
