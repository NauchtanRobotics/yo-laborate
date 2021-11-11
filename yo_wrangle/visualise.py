import numpy as np

from cv2 import cv2
from pathlib import Path
from typing import List, Optional


def draw_polygon_on_image(
    image_file: str,
    coords: List[List[float]],
    dst_path: Path = None,
    class_name: Optional[str] = None,
):
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
