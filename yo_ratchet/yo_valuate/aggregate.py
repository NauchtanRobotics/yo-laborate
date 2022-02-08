from pathlib import Path
from typing import Dict, List

import pandas

from yo_ratchet.yo_valuate.expected_values import (
    get_moving_average_for_combined_expectation_from_multiple_classes,
    COUNT_STR,
)


def get_max_object_intensity_by_grouping_variable(
    inferences_file: Path,
    class_ids: List[int],
    span: int,
    shape_file: Path,
    shape_file_grouping_column: str,
    shape_file_index_column: str = "Photo_Name",
    metric: str = "count",  # count | length | area
) -> Dict[str, float]:
    """
    Reads in the defects csv and filters for sealed images.
    Builds up a new dataframe by cycling through unique values of the <shape_file_grouping_column>
    and finds the expected value (count, length (diagonal) or area)
    of all instances for a given class_id in an image.

    Next, call a function that finds the moving average of the expected_value.
    and returns a sorted dict of the max(moving_average(expected_value))
    for each road. For example, if shape_file_grouping_column is a column containing
    a road name, the series returned may look like:

        {
            "Henry Rd": {"count": 4.4, "index": "Photo_257.jpg"},
            "Victoria St": {"count": 3.9, "index": "Photo_2007.jpg"}
        }
    This data is then optional printed as a pretty table.

    This process is the repeated for each class_id in class_ids.

    """
    if inferences_file.suffix == ".ai":
        use_cols = [0, 1, 2, 3, 4, 5, 10]
        if metric != COUNT_STR:
            print(
                "\nThis file contains polygons data so cannot calculate this metric."
                "\nCalculating the expected object 'count' instead."
            )
            metric = COUNT_STR
    else:
        use_cols = [0, 1, 2, 3, 4, 5, 6]

    df_inference = pandas.read_csv(
        filepath_or_buffer=str(inferences_file),
        header=None,
        sep=" ",
        usecols=use_cols,
    )
    if inferences_file.suffix == ".ai":
        df_inference.rename(
            columns={
                0: "Photo_Name",
                1: "class_id",
                10: "prob",
            },
            inplace=True,
        )
    else:
        df_inference.rename(
            columns={
                0: "Photo_Name",
                1: "class_id",
                2: "x_centre",
                3: "y_centre",
                4: "width",
                5: "height",
                6: "prob",
            },
            inplace=True,
        )

    df_shape_file = pandas.read_csv(filepath_or_buffer=str(shape_file))
    group_names = df_shape_file[shape_file_grouping_column].unique()

    value_results = index_results = {}
    for group_name in group_names:
        df = df_shape_file.loc[df_shape_file[shape_file_grouping_column] == group_name]
        if len(df) <= span:
            continue
        moving_average = (
            get_moving_average_for_combined_expectation_from_multiple_classes(
                df=df.copy(),
                df_inference=df_inference,
                class_ids=class_ids,
                span=span,
                metric=metric,
                index_column_name=shape_file_index_column,
            )
        )
        value_results[group_name] = moving_average.max(axis=0, skipna=True)
        index_results[group_name] = moving_average.idxmax(axis=0, skipna=True)

    value_results = dict(
        sorted(value_results.items(), key=lambda item: item[1], reverse=True)
    )
    value_results = {
        key: {"count": max_val, "index": index_results[key]}
        for key, max_val in value_results.items()
    }
    return value_results
