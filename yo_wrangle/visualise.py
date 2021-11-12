import numpy as np
import pandas
from cv2 import cv2
from pathlib import Path
from typing import List, Optional

from yo_wrangle.common import get_all_jpg_recursive, get_id_to_label_map


def draw_polygon_on_image(
    image_file: str,
    coords: List[List[float]],
    dst_path: Path = None,
    class_name: Optional[str] = None,
):
    """
    This function takes a copy of an image and draws a bounding box
    (polygon) according to the provided `coords` parameter.

    If a `dst_path` is provided, the resulting image will be saved.
    Otherwise, the image will be displayed in a pop up window.

    """
    image = cv2.imread(image_file)
    height, width, channels = image.shape

    polygon_1 = [[x * width, y * height] for x, y in coords]
    polygon_1 = np.array(polygon_1, np.int32).reshape((-1, 1, 2))
    is_closed = True
    thickness = 2
    cv2.polylines(image, [polygon_1], is_closed, (0, 255, 0), thickness)
    if class_name:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, class_name, (200, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, class_name, (203, 503), font, 4, (0, 0, 0), 2, cv2.LINE_AA)
    if dst_path:
        cv2.imwrite(str(dst_path), image)
    else:
        cv2.imshow("Un-transformed Bounding Box", image)
        cv2.waitKey()
        cv2.destroyAllWindows()


def save_bounding_boxes_on_images(
    images_root: Path,
    dst_root: Path,
    ai_file_path: Path,
    class_list_path: Path,
):
    """
    Save a copy of all images from images_root to dst_root with bounding boxes applied.
    Only one bounding box is drawn on each image in dst_root.  Filenames in dst_root
    include an index for the defect number.

    If more than one defect was found in an image, there will be multiple corresponding
    images in dst_root.

    """
    df = pandas.read_csv(
        filepath_or_buffer=ai_file_path,
        header=None,
        sep=" ",
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        names=[
            "Photo_Name",
            "Class_ID",
            "x1",
            "y1",
            "x2",
            "y2",
            "x3",
            "y3",
            "x4",
            "y4",
        ],
    )
    images_with_defects = df["Photo_Name"].unique()
    print("\nCount images with defects = ", len(images_with_defects))
    assert dst_root.exists() is False, "Destination directory already exists"
    dst_root.mkdir(parents=True)

    id_to_class_name_map = get_id_to_label_map(class_name_list_path=class_list_path)

    for img_path in get_all_jpg_recursive(img_root=images_root):
        photo_name = img_path.name
        image_data = df.loc[df["Photo_Name"] == photo_name].reset_index()
        if len(image_data) == 0:
            continue
        for index, row in image_data.iterrows():
            class_id = row["Class_ID"]
            # if int(class_id) in [7, 10]:
            #     continue
            class_name = id_to_class_name_map[class_id]
            series = row[
                [
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "x3",
                    "y3",
                    "x4",
                    "y4",
                ]
            ]
            bounding_box_coords = [
                [series["x1"], series["y1"]],
                [series["x2"], series["y2"]],
                [series["x3"], series["y3"]],
                [series["x4"], series["y4"]],
            ]

            dst_path = dst_root / f"{img_path.stem}_{index}{img_path.suffix}"
            draw_polygon_on_image(
                image_file=str(img_path),
                coords=bounding_box_coords,
                dst_path=dst_path,
                class_name=class_name
            )
