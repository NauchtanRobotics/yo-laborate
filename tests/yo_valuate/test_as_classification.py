from pathlib import Path

import numpy
import pandas
import pytest

from yo_ratchet.yo_valuate.as_classification import (
    optimise_model_binary_metrics_for_groups,
    _optimise_analyse_model_binary_metrics,
)

ROOT_TEST_DATA = Path(__file__).parent.parent / "test_data"
CLASSES_MAP = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
    4: "AA",
    5: "AB",
    6: "AC",
    7: "AD",
    8: "AE",
    9: "AF",
    10: "BA",
    11: "BB",
    12: "BC",
    13: "BD",
    14: "BE",
    15: "BF",
    16: "CA",
    17: "CB",
    18: "CC",
    19: "CD",
    20: "CE",
}


def test_analyse_model_binary_metrics_for_groups():
    images_root = ROOT_TEST_DATA / "classification" / "images"
    root_ground_truths = ROOT_TEST_DATA / "filter_yolo"
    root_inferences = ROOT_TEST_DATA / "classification" / "labels"

    df = optimise_model_binary_metrics_for_groups(
        images_root=images_root,
        root_ground_truths=root_ground_truths,
        root_inferred_bounding_boxes=root_inferences,
        classes_map=CLASSES_MAP,
        groupings={
            "Risk Defects": [3, 4],
            "Cracking": [0, 1, 2, 16],
        },
    )
    assert isinstance(df, pandas.DataFrame)
    assert "Risk Defects" in list(df)
    assert "Cracking" in list(df)
    assert "P" in df.index
    assert "R" in df.index
    assert "F1" in df.index
    assert "@conf" in df.index


@pytest.fixture
def dummy_df():
    results_dict = {
        "001": {
            "actual_classifications": numpy.array([False, True, True]),
            "inferred_classifications": [True, True, True],
            "confidence": [0.1, 0.3, 0.4],
        },
        "002": {
            "actual_classifications": numpy.array([True, False, True]),
            "inferred_classifications": [True, True, True],
            "confidence": [0.5, 0.3, 0.4],
        },
        "003": {
            "actual_classifications": numpy.array([False, False, False]),
            "inferred_classifications": [True, True, True],
            "confidence": [0.15, 0.2, 0.2],
        },
        "004": {
            "actual_classifications": numpy.array([False, False, False]),
            "inferred_classifications": [False, False, False],
            "confidence": [0.0, 0.0, 0.0],
        },
        "005": {
            "actual_classifications": numpy.array([False, True, False]),
            "inferred_classifications": [False, False, False],
            "confidence": [0.0, 0.0, 0.0],
        },
    }
    df = pandas.DataFrame(results_dict)
    df = df.transpose()
    return df


def test__optimise_analyse_model_binary_metrics(dummy_df):
    classes_map = {0: "Casper", 1: "Friendly", 2: "Ghost"}
    res = _optimise_analyse_model_binary_metrics(
        df=dummy_df, classes_map=classes_map, print_first_n=3
    )
    assert res == {
        "Casper": {"@conf": "0.16", "F1": "1.00", "P": "1.00", "R": "1.00"},
        "Friendly": {"@conf": "0.25", "F1": "0.50", "P": "0.50", "R": "0.50"},
        "Ghost": {"@conf": "0.25", "F1": "1.00", "P": "1.00", "R": "1.00"},
    }
