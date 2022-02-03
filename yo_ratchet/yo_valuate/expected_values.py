import math
import pandas
from typing import Optional, List

AREA_STR = "area"
LENGTH_STR = "length"
COUNT_STR = "count"


def append_expected_value_column_to_df(
    df: pandas.DataFrame,
    df_inferences: pandas.DataFrame,
    class_id: int,
    metric: str = COUNT_STR,  # count | length | area
    image_reference_key: str = "Photo_Name",
    result_column_name: str = "expected_value",
) -> pandas.DataFrame:
    """
    Given any pandas.DataFrame, df, this function appends a column named
    <result_column_name> which contains the probabilistic expected count, length
    or area of object detections that correspond to a given class id.

    A pandas.DataFrame `df_inferences` must be provided to supply to object
    detection inferences where this data frame has a column named <image_reference_key>
    which has values corresponding a column by the same name in df.
    There is a one to many (0, 1 or more) relationship between the
    df and df_inferences when joined on this key.

    """
    df[result_column_name] = df.apply(
        lambda x: get_expected_value_for_image(
            df_inferences=df_inferences,
            image_reference=x[image_reference_key],
            class_id=class_id,
            image_reference_key=image_reference_key,
            metric=metric,
        ),
        axis=1,
    )
    return df


def get_expected_value_for_image(
    df_inferences: pandas.DataFrame,
    image_reference: str,
    class_id: int,
    image_reference_key: str = "Photo_Name",
    metric: str = COUNT_STR,
) -> float:
    """
    Returns an expected object count, diagonal length, or area for a given
    image name and class id. An expected value takes into account the
    confidence of each object detection when calculating the sum.

    The columns of the dataframe must be like a yolo annotation preceded by
    the image name, i.e. must contain data in this order::

        photo reference, class_id, x_centre, y_centre, width, height, probability

    :param df_inferences:
    :param image_reference: The image for which to find the total expected value
    :param class_id: ID of the class for which the
    :param image_reference_key: name of pandas column which contains image name.
    :param metric: one of "count" | "length" | "area". Determine how the expected
                   value is calculated.
    :return: expect value for the select metric.

    """
    df_inferences = df_inferences.loc[
        df_inferences[image_reference_key] == image_reference
    ]
    df = df_inferences.copy()
    df.drop(image_reference_key, axis=1, inplace=True)
    accumulator = 0.0
    for index, row in df.iterrows():
        [class_str, _, _, width, height, prob] = [val for _, val in row.items()]
        if int(class_str) != class_id:
            continue
        if metric == COUNT_STR:
            accumulator += float(prob)
        elif metric == LENGTH_STR:
            diagonal = math.pow(float(width), 2) + math.pow(float(height), 2)
            diagonal = math.sqrt(diagonal)
            accumulator += float(prob) * diagonal
        elif metric == AREA_STR:
            area = float(width) * float(height)
            accumulator += float(prob) * area
    return accumulator


def get_moving_average_for_target_column(
    df: pandas.DataFrame,
    span: int,
    index_column: Optional[str] = None,
    target_column: str = "expected_value",
) -> pandas.Series:
    """
    Provided with a pandas dataframe, df, which has a column named <target_column>
    - performs a centred moving average calculation using `span` for the
    window size.

    Returns a pandas series of the moving average values, with the index_column
    parameter determining which column of `df` that is applied as the series index.

    The series is then trimmed to remove the first and last `(span-1)/2` elements
    which are averaged over less than span elements.

    n.b. This function has only validated for span as an odd number.

    """
    moving_average = df[target_column].rolling(window=span, center=True).mean()
    rows_to_trim = int((span - 1) / 2)
    if index_column:
        moving_average.index = df[index_column]
    if len(df) >= span:
        moving_average = moving_average[rows_to_trim:-rows_to_trim]
    return moving_average


def get_moving_average_for_combined_expectation_from_multiple_classes(
    df: pandas.DataFrame,
    df_inference: pandas.DataFrame,
    class_ids: List[int],
    span: int,
    metric: str = COUNT_STR,
    index_column_name: Optional[str] = "Photo_Name",
    image_reference_key: str = "Photo_Name",
) -> pandas.Series:
    result_columns = []
    for class_id in class_ids:
        target_column = f"class_{str(class_id)}"
        result_columns.append(target_column)
        df = append_expected_value_column_to_df(
            df=df,
            df_inferences=df_inference,
            class_id=class_id,
            metric=metric,
            image_reference_key=image_reference_key,
            result_column_name=target_column,
        )
    result = "combined_expectation"
    df[result] = df[result_columns].sum(axis=1, skipna=True)
    moving_average = get_moving_average_for_target_column(
        df=df, span=span, index_column=index_column_name, target_column=result
    )
    return moving_average
