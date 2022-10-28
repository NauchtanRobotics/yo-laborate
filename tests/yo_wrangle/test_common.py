from pathlib import Path

from yo_ratchet.yo_wrangle.common import get_subsets_included


def test_get_subsets_nothing_excluded():
    test_data_repo = Path(__file__).parent.parent / "test_data/sample_data_repository"
    assert test_data_repo.exists()
    subsets = get_subsets_included(test_data_repo)
    assert subsets == [test_data_repo / "Subset_1",  test_data_repo / "Subset_2"]


def test_get_subsets_with_subsets_excluded():
    test_data_repo = Path(__file__).parent.parent / "test_data/sample_data_repository_with_excluded"
    assert test_data_repo.exists()
    subsets = get_subsets_included(test_data_repo)
    assert subsets == [test_data_repo / "Subset_2"]
