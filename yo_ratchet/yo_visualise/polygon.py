from typing import List, Tuple


class PerspectiveCoordinateFactors:
    def __init__(self):
        self.top_left_x = 0.3
        self.top_right_x = 0.7
        self.bottom_left_x = 0
        self.bottom_right_x = 1.0

        self.top_left_y = 0.50
        self.top_right_y = 0.50
        self.bottom_left_y = 1.0
        self.bottom_right_y = 1.0


def reverse_transform_polygon_coordinates(
    polygon: Tuple[float, float, float, float, float, float, float, float],
    perspective_params: PerspectiveCoordinateFactors,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Takes the coordinates for bounding box polygon (normalised) for
    defects detected on a transformed image, then reverse transforms
    these coordinates so that the will roughly represent the polygon
    in the original image space.

    Assume that the original transform had the top 50% of the image
    cropped out, and the perspective transform then trimmed 20% in
    x direction from either edge of the original image at the y=0.5
    mark, i.e. 0.25 < x < 0.75. In actuality, datasets will be trim-
    med anywhere from 0.1 < x < 0.65 and 0.3 < x < 0.85.


    Assumes y-bottom-left = y-bottom-right = 1.0 (although
    some datasets this can be 0.95 or even 0.9 to remove bumper)
    and x-bottom-left = x-bottom-right = 1.0 (mostly true).

    """
    trim_y_min = perspective_params.top_left_y
    trim_y_max = perspective_params.bottom_left_y

    trim_x_min = perspective_params.top_left_x  # 0.25
    trim_x_max = 1 - perspective_params.top_right_x
    trim_mid_point = (
        perspective_params.top_left_x + perspective_params.top_right_x
    ) / 2.0

    min_x, min_y, min_x, max_y, max_x, max_y, max_x, min_y = polygon

    x_trim_top_left = (1 - min_y) * trim_x_min
    x_trim_top_right = (1 - min_y) * trim_x_max
    x_trim_bottom_left = (1 - max_y) * trim_x_min
    x_trim_bottom_right = (1 - max_y) * trim_x_max

    if min_x < 0.5:
        min_x_top = clamp_float_0_1_range(
            trim_mid_point
            - (0.5 - min_x) * (trim_mid_point - x_trim_top_left) / trim_mid_point
        )
        min_x_bottom = clamp_float_0_1_range(
            trim_mid_point
            - (0.5 - min_x) * (trim_mid_point - x_trim_bottom_left) / trim_mid_point
        )
    else:
        min_x_top = clamp_float_0_1_range(
            0.5 + (min_x - 0.5) * (trim_mid_point - x_trim_top_right) / trim_mid_point
        )
        min_x_bottom = clamp_float_0_1_range(
            0.5
            + (min_x - 0.5) * (trim_mid_point - x_trim_bottom_right) / trim_mid_point
        )

    if max_x < 0.5:
        max_x_top = clamp_float_0_1_range(
            trim_mid_point
            - (0.5 - max_x) * (trim_mid_point - x_trim_top_left) / trim_mid_point
        )
        max_x_bottom = clamp_float_0_1_range(
            trim_mid_point
            - (0.5 - max_x) * (trim_mid_point - x_trim_bottom_left) / trim_mid_point
        )
    else:
        max_x_top = clamp_float_0_1_range(
            0.5 + (max_x - 0.5) * (trim_mid_point - x_trim_top_right) / trim_mid_point
        )
        max_x_bottom = clamp_float_0_1_range(
            0.5
            + (max_x - 0.5) * (trim_mid_point - x_trim_bottom_right) / trim_mid_point
        )

    scale_y = trim_y_max - trim_y_min
    min_y = clamp_float_0_1_range(
        trim_y_min
        + min_y * scale_y * min_y
        + (1 - min_y) * min_y * min_y * scale_y * 0.5
    )
    max_y = clamp_float_0_1_range(
        trim_y_min
        + max_y * scale_y * max_y
        + (1 - max_y) * max_y * max_y * scale_y * 0.5
    )

    return min_x_top, min_y, min_x_bottom, max_y, max_x_bottom, max_y, max_x_top, min_y


def clamp_float_0_1_range(num, eps: float = 1e-7):
    return max(min(num, 1 - eps), 0 + eps)


def convert_to_polygons(lines: List) -> List[List]:
    """
    Converts a line from a yolo annotations file from yolo format to
    polygon data that can be used by

    """
    new_lines = []
    for line in lines:
        class_id = line[0]
        yolo_coords = [float(element) for element in line[1:5]]
        if len(line) == 6:
            probability = line[5]
        else:
            probability = None
        min_x = yolo_coords[0] - yolo_coords[2] / 2
        max_x = yolo_coords[0] + yolo_coords[2] / 2
        min_y = yolo_coords[1] - yolo_coords[3] / 2
        max_y = yolo_coords[1] + yolo_coords[3] / 2
        polygon = (min_x, min_y, min_x, max_y, max_x, max_y, max_x, min_y)
        polygon = [str(element) for element in polygon]
        new_line = [class_id]
        new_line.extend(polygon)
        if probability:
            new_line.append(probability)
        new_lines.append(new_line)
    return new_lines
