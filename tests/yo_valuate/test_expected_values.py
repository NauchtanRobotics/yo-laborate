import pandas
import pytest
from pandas import testing as pandas_testing


from yo_ratchet.yo_valuate.expected_values import (
    get_moving_average_for_target_column,
    append_expected_value_column_to_df,
    get_expected_value_for_image,
    COUNT_STR,
    AREA_STR,
    LENGTH_STR,
    get_moving_average_for_combined_expectation_from_multiple_classes,
)


def test_get_moving_average_of_expected_value():
    index_name = "Position"
    expected_series = pandas.Series(data=[0.55, 0.2, 0.1], index=[15, 25, 35])
    expected_series.index.name = index_name
    expected_series.name = "expected_value"

    span = 3
    index = ["photo_1.jpg", "photo_2.jpg", "photo_3.jpg", "photo_4.jpg", "photo_5.jpg"]
    data = {
        "Position": [5, 15, 25, 35, 45],
        "expected_value": [1.2, 0.45, 0, 0.15, 0.15],
    }
    df = pandas.DataFrame(data=data)
    df.index = index
    moving_average = get_moving_average_for_target_column(
        df=df, span=span, index_column=index_name
    )

    assert isinstance(moving_average, pandas.Series)
    assert len(moving_average) == len(df) - span + 1
    pandas_testing.assert_series_equal(moving_average, expected_series)


@pytest.fixture
def custom_df():
    data = {
        "Photo_Name": [
            "photo_1.jpg",
            "photo_2.jpg",
            "photo_3.jpg",
            "photo_4.jpg",
            "photo_5.jpg",
        ],
        "Sealed": [0, 1, 0, 1, 0],
        "Chainage": [5, 15, 25, 35, 45],
    }
    df = pandas.DataFrame(data=data)
    return df


@pytest.fixture
def inference_df():
    photo_names = [
        "photo_1.jpg",
        "photo_1.jpg",
        "photo_1.jpg",
        "photo_2.jpg",
        "photo_2.jpg",
    ]
    class_ids = [1, 2, 2, 1, 2]
    x_centres = [0.5, 0.5, 0.5, 0.5, 0.5]
    y_centres = [0.5, 0.5, 0.5, 0.5, 0.5]
    widths = [0.1, 0.1, 0.1, 0.2, 0.2]
    heights = [0.1, 0.1, 0.1, 0.2, 0.2]
    probs = [0.1, 0.2, 0.3, 0.4, 0.5]

    data = {
        "Photo_Name": photo_names,
        "class_id": class_ids,
        "x_centre": x_centres,
        "y_centre": y_centres,
        "width": widths,
        "height": heights,
        "prob": probs,
    }
    df_inference = pandas.DataFrame(data=data)
    return df_inference


def test_append_expected_value_column_to_df(custom_df, inference_df):
    result = append_expected_value_column_to_df(
        df=custom_df,
        df_inferences=inference_df,
        class_id=2,
        metric=COUNT_STR,
        image_reference_key="Photo_Name",
    )
    assert result["expected_value"].tolist() == [0.5, 0.5, 0.0, 0.0, 0.0]


def test_get_expected_value_for_image(inference_df):
    assert (
        get_expected_value_for_image(
            df_inferences=inference_df,
            image_reference="photo_1.jpg",
            class_id=1,
            metric=COUNT_STR,
        )
        == 0.1
    )

    assert (
        get_expected_value_for_image(
            df_inferences=inference_df,
            image_reference="photo_1.jpg",
            class_id=2,
            metric=COUNT_STR,
        )
        == 0.5
    )

    assert (
        round(
            get_expected_value_for_image(
                df_inferences=inference_df,
                image_reference="photo_1.jpg",
                class_id=1,
                metric=LENGTH_STR,
            ),
            7,
        )
        == 0.0141421
    )

    assert (
        round(
            get_expected_value_for_image(
                df_inferences=inference_df,
                image_reference="photo_1.jpg",
                class_id=2,
                metric=LENGTH_STR,
            ),
            7,
        )
        == 0.0707107
    )

    assert (
        round(
            get_expected_value_for_image(
                df_inferences=inference_df,
                image_reference="photo_1.jpg",
                class_id=2,
                metric=AREA_STR,
            ),
            7,
        )
        == 0.005
    )

    assert (
        get_expected_value_for_image(
            df_inferences=inference_df,
            image_reference="photo_2.jpg",
            class_id=1,
            metric=COUNT_STR,
        )
        == 0.4
    )


@pytest.fixture
def inference_df_2():
    photo_names = [
        "photo_1.jpg",
        "photo_1.jpg",
        "photo_2.jpg",
        "photo_2.jpg",
        "photo_3.jpg",
    ]
    class_ids = [
        1,
        1,
        1,
        1,
        1,
    ]
    x_centres = [0.5, 0.5, 0.5, 0.5, 0.5]
    y_centres = [0.5, 0.5, 0.5, 0.5, 0.5]
    widths = [0.1, 0.1, 0.1, 0.2, 0.2]
    heights = [0.1, 0.1, 0.1, 0.2, 0.2]
    probs = [0.1, 0.2, 0.3, 0.4, 0.5]

    data = {
        "Photo_Name": photo_names,
        "class_id": class_ids,
        "x_centre": x_centres,
        "y_centre": y_centres,
        "width": widths,
        "height": heights,
        "prob": probs,
    }
    df_inference = pandas.DataFrame(data=data)
    return df_inference


def test_integrated(custom_df, inference_df_2):
    df = append_expected_value_column_to_df(
        df=custom_df,
        df_inferences=inference_df_2,
        class_id=1,
        metric=COUNT_STR,
        image_reference_key="Photo_Name",
    )
    # Check that expected values are as expected
    result = df["expected_value"].tolist()
    result = [round(val, 5) for val in result]
    assert result == [0.3, 0.7, 0.5, 0.0, 0.0]

    moving_average = get_moving_average_for_target_column(df=df, span=3)
    assert isinstance(moving_average, pandas.Series)
    assert len(moving_average) == 3
    result = [round(el, 4) for el in moving_average.values]
    assert result == [0.5, 0.4, 0.1667]


def test_get_moving_average_combined_score(custom_df, inference_df_2):
    class_ids = [0, 1, 2, 3]
    df = custom_df.copy()
    span = 3
    moving_average = get_moving_average_for_combined_expectation_from_multiple_classes(
        df=df, df_inference=inference_df_2, class_ids=class_ids, span=span
    )
    result = [round(el, 4) for el in moving_average.tolist()]
    assert result == [0.5, 0.4, 0.1667]
