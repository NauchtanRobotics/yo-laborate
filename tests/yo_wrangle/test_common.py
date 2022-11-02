from pathlib import Path

import pytest

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
    """
    This selected config file has a max count of 3 so all 3 of a possible 3 model paths are returned.

    """
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
    """
    This selected config file has a max count of 2 so only the first 2 of a possible 3 model paths are returned.

    """
    models = get_implicit_model_paths(
        base_dir=TEST_DATA_PATH / "config_file_model_version",
        dataset_identifier="SEALED"
    )
    assert models == [
        TEST_DATA_PATH / "yolov5/runs/train" / "6.1.1" / "weights" / "best.pt",
        TEST_DATA_PATH / "yolov5/runs/train" / "6.1.2" / "weights" / "best.pt",
    ]


def test_get_implicit_model_paths_config_file_uses_tilde():
    """
    The selected config file uses a `~` in nominating the ENSEMBLE_MODEL_ROOT.
    This doesn't actually test whether the tilde resolves correctly, but assuming it does,
    the FUT is expected to raise an exception because the path listed in the config file
    for the ENSEMBLE_MODEL_ROOT does not exist (unless ~/36.1 exists!)

    This test is to ensure that the error message raised is helpful to a user in correctly
    diagnosing that it was the resolved model file path not existing which caused
    the code to fail.

    """
    with pytest.raises(Exception) as ex_info:
        models = get_implicit_model_paths(
            base_dir=TEST_DATA_PATH / "config_file_ensemble_model_root_uses_tilde",
            dataset_identifier="SEALED"
        )
    str_info = str(ex_info.value)
    assert "Path does not exist" in str_info
