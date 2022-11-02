from pathlib import Path

from yo_ratchet.yo_wrangle.common import get_subsets_included, get_implicit_model_paths

TEST_DATA_PATH = Path(__file__).parent.parent / "test_data"


def test_get_subsets_nothing_excluded():
    test_data_repo = TEST_DATA_PATH / "sample_data_repository"
    assert test_data_repo.exists()
    subsets = get_subsets_included(test_data_repo)
    assert subsets == [test_data_repo / "Subset_1",  test_data_repo / "Subset_2"]


def test_get_subsets_with_subsets_excluded():
    test_data_repo = TEST_DATA_PATH / "sample_data_repository_with_excluded"
    assert test_data_repo.exists()
    subsets = get_subsets_included(test_data_repo)
    assert subsets == [test_data_repo / "Subset_2"]


def test_get_implicit_model_paths_ensemble_model_root():
    models = get_implicit_model_paths(
        base_dir=TEST_DATA_PATH / "config_file_ensemble_model_root",
        dataset_identifier="SEALED"
    )
    assert models == [
        TEST_DATA_PATH / "model_36.1" / "36.1.1" / "weights" / "best.pt",
        TEST_DATA_PATH / "model_36.1" / "36.1.2" / "weights" / "best.pt",
        TEST_DATA_PATH / "model_36.1" / "36.1.3" / "weights" / "best.pt"
    ]


def test_get_implicit_model_paths_model_version_max_count():
    models = get_implicit_model_paths(
        base_dir=TEST_DATA_PATH / "config_file_model_version",
        dataset_identifier="SEALED"
    )
    assert models == [
        TEST_DATA_PATH / "yolov5/runs/train" / "6.1.1" / "weights" / "best.pt",
        TEST_DATA_PATH / "yolov5/runs/train" / "6.1.2" / "weights" / "best.pt",
    ]
