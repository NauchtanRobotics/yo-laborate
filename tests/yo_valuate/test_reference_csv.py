import pandas
import pytest
from pandas._testing import assert_frame_equal
from pathlib import Path
from typing import Dict, List

from yo_valuate.reference_csv import (
    _get_group_memberships_from_dataframe,
    get_group_memberships_truths,
    get_group_membership_inferences,
    get_thresholds,
    get_actual_vs_inferred_df,
    get_classification_performance,
    get_severity_dict,
)

ROOT_TEST_DATA = Path(__file__).parent.parent / "test_data"


@pytest.fixture
def test_csv_group_filters() -> Dict[str, List[str]]:
    return {
        "People": ["Bob", "fred"],
        "Beverages": ["Wine", "Beer"],
    }


@pytest.fixture
def test_df():
    data = {
        "Photo_NameZ": ["Photo_1", "Photo_2", "Photo_3", "Photo_4"],
        "Final_Remedy": [
            "Fred  Bob  Wine",
            "Fred",
            "  Bob",
            "  bob wine",
        ],
    }
    df = pandas.DataFrame(data)
    return df


def test__get_group_memberships(test_csv_group_filters, test_df):
    results = _get_group_memberships_from_dataframe(
        df=test_df,
        csv_group_mappings=test_csv_group_filters,
        image_key="Photo_NameZ",
        classifications_key="Final_Remedy",
    )
    expected_results = {
        "Photo_1": list({"People", "Beverages"}),
        "Photo_2": ["People"],
        "Photo_3": ["People"],
        "Photo_4": list({"People", "Beverages"}),
    }
    assert results == expected_results


def test_get_group_memberships_truths_mocked_read(
    mocker, test_df, test_csv_group_filters
):
    mocker.patch("pandas.read_csv", return_value=test_df)
    mocker.patch("yo_valuate.reference_csv.get_severity_dict", return_value={
        "Photo_1": 8,
        "Photo_2": 8,
        "Photo_3": 8,
        "Photo_4": 8
    })
    expected_truths = {  # [has_people, has_beverage]
        "Photo_1": [True, True],
        "Photo_2": [True, False],
        "Photo_3": [True, False],
        "Photo_4": [True, True],
    }
    result = get_group_memberships_truths(
        truths_csv=Path(),
        csv_group_filters=test_csv_group_filters,
        image_key="Photo_NameZ",
        classifications_key="Final_Remedy",
        severity_key="D2_Side",
    )
    assert result == expected_truths


@pytest.fixture
def expected_truths():
    return {  # [has_people, has_beverage]
        "Photo_1.jpg": [True, False],
        "Photo_2.jpg": [True, False],
        "Photo_3.jpg": [True, True],
        "Photo_4.jpg": [False, True],
    }


def test_get_group_memberships_truths(test_df, test_csv_group_filters, expected_truths):
    truths_csv = ROOT_TEST_DATA / "classification" / "truth.csv"
    result = get_group_memberships_truths(
        truths_csv=truths_csv,
        csv_group_filters=test_csv_group_filters,
        image_key="Photo_Name",
        classifications_key="D2_Remedy",
        severity_key="D2_Side",
    )
    assert result == expected_truths


@pytest.fixture
def test_yolo_group_filters() -> Dict[str, List[int]]:
    return {
        "People": [0, 1, 2],
        "Beverages": [3, 4, 5],
    }


@pytest.fixture
def thresholds() -> Dict[int, float]:
    return {
        0: 0.1,
        1: 0.11,
        2: 0.12,
        3: 0.22,
        4: 0.21,
        5: 0.20,
    }


@pytest.fixture
def expected_inferences():
    return {  # [has_people, has_beverage]
        "Photo_1.jpg": [False, False],
        "Photo_2.jpg": [True, False],
        "Photo_3.jpg": [False, False],
        "Photo_4.jpg": [False, False],
    }


def test_get_group_membership_inferences(
    test_csv_group_filters, test_yolo_group_filters, thresholds, expected_inferences
):
    images_root = ROOT_TEST_DATA / "classification" / "images"
    root_inferences = ROOT_TEST_DATA / "classification" / "labels"

    result = get_group_membership_inferences(
        images_root=images_root,
        root_inferred_bounding_boxes=root_inferences,
        csv_group_filters=test_csv_group_filters,
        yolo_group_filters=test_yolo_group_filters,
        thresholds=thresholds,
    )
    assert result == expected_inferences


@pytest.fixture
def classes_info() -> Dict[str, Dict]:
    return {
        "0": {"label": "People0", "min_prob": 0.1},
        "1": {"label": "People1", "min_prob": 0.11},
        "2": {"label": "People2", "min_prob": 0.12},
        "3": {"label": "People3", "min_prob": 0.22},
        "4": {"label": "Beverage1", "min_prob": 0.21},
        "5": {"label": "Beverage2", "min_prob": 0.20},
    }


def test_get_thresholds(classes_info, thresholds):
    thresholds = get_thresholds(classes_info=classes_info)
    assert thresholds == thresholds


def test_get_actual_vs_inferred_df(
    classes_info,
    test_csv_group_filters,
    test_yolo_group_filters,
    expected_inferences,
    expected_truths,
):
    images_root = ROOT_TEST_DATA / "classification" / "images"
    root_inferences = ROOT_TEST_DATA / "classification" / "labels"
    truths_csv = ROOT_TEST_DATA / "classification" / "truth.csv"
    df = get_actual_vs_inferred_df(
        images_root=images_root,
        truths_csv=truths_csv,
        root_inferred_bounding_boxes=root_inferences,
        csv_group_filters=test_csv_group_filters,
        yolo_group_filters=test_yolo_group_filters,
        classes_info=classes_info,
        image_key="Photo_Name",
        classifications_key="D2_Remedy",
        severity_key="D2_Side",
    )
    indices = sorted(expected_inferences.keys())
    expected_result = pandas.DataFrame(
        {
            "index": indices,
            "actual_classifications": [expected_truths[index] for index in indices],
            "inferred_classifications": [
                expected_inferences[index] for index in indices
            ],
        }
    )
    expected_result.set_index(keys="index", drop=True, inplace=True)
    expected_result.index.names = [None]
    assert_frame_equal(df, expected_result)


def test_get_classification_performance(
    classes_info,
    test_csv_group_filters,
    test_yolo_group_filters,
    expected_inferences,
    expected_truths,
):
    images_root = ROOT_TEST_DATA / "classification" / "images"
    root_inferences = ROOT_TEST_DATA / "classification" / "labels"
    truths_csv = ROOT_TEST_DATA / "classification" / "truth.csv"
    df = get_classification_performance(
        images_root=images_root,
        truths_csv=truths_csv,
        root_inferred_bounding_boxes=root_inferences,
        csv_group_filters=test_csv_group_filters,
        yolo_group_filters=test_yolo_group_filters,
        classes_info=classes_info,
        image_key="Photo_Name",
        classifications_key="D2_Remedy",
        severity_key="D2_Side",
        severity_threshold=8,
    )
    assert isinstance(df, pandas.DataFrame)
    res_dict = df.to_dict()
    expected_result = {
        "People": {"P": "1.00", "R": "0.33", "F1": "0.50"},
        "Beverages": {"P": "0.00", "R": "0.00", "F1": "0.00"},
    }
    assert res_dict == expected_result


def test_get_severity_dict():
    truths_csv = Path(__file__).parents[2] / "tests/test_data/classification/truth.csv"
    get_severity_dict(
        truths_csv=truths_csv, field_for_severity="D2_Side", field_for_key="Photo_Name", default_severity=10
    )
