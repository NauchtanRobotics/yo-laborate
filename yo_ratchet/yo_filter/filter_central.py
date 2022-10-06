import cv2
from pathlib import Path
from typing import List, Optional, Tuple

from yo_ratchet.yo_wrangle.common import get_all_jpg_recursive, get_all_txt_recursive

WEDGE_GRADIENTS = (-2.0, 2.0)
WEDGE_CONSTANTS = (1.0, -1.0)
YOLO_ANNOTATIONS_FOLDER_NAME = "YOLO_darknet"


def calculate_wedge_envelop(
    x_apex: float,
    y_apex: float,
):
    m1 = (1.0 - y_apex) / (0.0 - x_apex)
    m2 = (1.0 - y_apex) / (1.0 - x_apex)

    c1 = 1.0
    c2 = 1.0 - m2 * 1.0

    return (c1, c2), (m1, m2)


def defect_is_central(
    yolo_coordinates: List[float],
    wedge_gradients: Optional[Tuple[float, float]] = None,
    wedge_constants: Optional[Tuple[float, float]] = None,
) -> bool:
    """
    Function to check if the centre of a bounding box around a defects
    is within the centre triangle of an image frame; i.e. excluding the
    top left and top right corner of the image frame::

        +------------------+
        |        /\        |
        |      /    \      |
        |    /        \    |
        |  /     X      \  |
        |/                \|
        +------------------+

    """
    if wedge_constants is None:
        c1, c2 = WEDGE_CONSTANTS[0], WEDGE_CONSTANTS[1]
    else:
        c1, c2 = wedge_constants[0], wedge_constants[1]

    if wedge_gradients is None:
        m1, m2 = WEDGE_GRADIENTS[0], WEDGE_GRADIENTS[1]
    else:
        m1, m2 = wedge_gradients[0], wedge_gradients[1]

    x_centroid = yolo_coordinates[0]
    y_centroid = yolo_coordinates[1]
    if x_centroid < 0.5:
        y_envelope = x_centroid * m1 + c1
    else:
        y_envelope = x_centroid * m2 + c2

    if y_centroid > y_envelope:
        return True
    else:
        return False


def has_d40_in_central_triangle(file_path: Path):
    """
    Inspects the YOLO_darknet annotations file for a single image
    and returns True if a D40 defect is found in the central triangle,
    or false otherwise.

    """
    with open(file_path) as f:
        lines = f.readlines()
    has_d40 = False
    for line in lines:
        line_list = line.strip().split(" ")
        class_id = line_list[0]

        if class_id == "3":
            coords = line_list[1:]
            yolo_coordinates = [float(x) for x in coords if x]
            has_d40 = defect_is_central(yolo_coordinates=yolo_coordinates)
            if has_d40:
                break
    return has_d40


def get_images_names_having_central_d40_defects(darknet_root: Path) -> List[str]:
    """
    Purpose is to iterate through a YOLO_darknet directory looking at the
    *.txt files and get a list image filenames that have d40 defects in the central
    triangle which omits defects in the top left and top right corner of
    the image frame.

    """
    images_with_central_d40 = []
    all_annotations_paths = sorted(list(darknet_root.rglob("*.txt")))
    for file_path in all_annotations_paths:
        if has_d40_in_central_triangle(file_path):
            images_with_central_d40.append(f"{file_path.stem}.jpg")

    return images_with_central_d40


def list_images_in_dir_with_probable_potholes(
    images_root: str,
    darknet_root: str,
):
    """
    Returns a list of images names based on the annotations filenames where
    the annotations file has at least one D40 defect (class_id == 3) in the
    central triangle of the image frame.

    """
    images_with_central_d40 = get_images_names_having_central_d40_defects(
        darknet_root=Path(darknet_root)
    )
    image_paths = list(get_all_jpg_recursive(img_root=Path(images_root)))
    image_paths = [
        img_path for img_path in image_paths if img_path.name in images_with_central_d40
    ]
    return image_paths


def test_copy_resize_images_in_dir_with_probable_potholes():
    """
    Function to select images with probable potholes (in the central triangle)

    Copy and resize them to 360 x 360 for demonstration purposes.

    """
    images_root = "/home/david/addn_repos/rddc2020/yolov5/inference/output"
    darknet_root = (
        "/home/david/RACAS/boosted/600_x_600/unmasked/RACAS_Isaac_2021/YOLO_darknet"
    )
    copy_to_dir = "/home/david/RACAS/boosted/600_x_600/unmasked/RACAS_Isaac_2021_boxed"

    copy_to_dir = Path(copy_to_dir)
    copy_to_dir.mkdir(exist_ok=True)

    image_paths = list_images_in_dir_with_probable_potholes(
        images_root=images_root,
        darknet_root=darknet_root,
    )

    for image_path in image_paths:
        dst_path = copy_to_dir / image_path.name
        image_cv2 = cv2.imread(str(image_path))
        image_cv2 = cv2.resize(image_cv2, (350, 350))
        cv2.imwrite(
            filename=str(dst_path),
            img=image_cv2,
            params=[int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )
    print()


def filter_top_left_and_right_defects_from_annotations(
    src_annotations_root: Path,
    dst_annotations_root: Path,
):
    dst_annotations_root.mkdir(exist_ok=True)
    for annotation_path in get_all_txt_recursive(root_dir=src_annotations_root):
        with open(annotation_path, "r") as src_file:
            lines = src_file.readlines()
        new_lines = []
        for line in lines:
            coordinates = line.strip("\n").split(" ")[1:5]
            coordinates = [float(coord) for coord in coordinates]
            if defect_is_central(
                yolo_coordinates=coordinates, wedge_gradients=None, wedge_constants=None
            ):
                new_lines.append(line)

        if len(new_lines) == 0:
            continue

        dst_file_path = dst_annotations_root / annotation_path.name
        with open(dst_file_path, "w") as dst_file:
            dst_file.writelines(new_lines)


def test_filter_top_left_and_right_defects_from_annotations():
    src_annotations_root = Path("")
    dst_annotations_root = Path("")
    filter_top_left_and_right_defects_from_annotations(
        src_annotations_root=src_annotations_root,
        dst_annotations_root=dst_annotations_root,
    )
