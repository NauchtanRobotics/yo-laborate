from pathlib import Path
from typing import List

import numpy
import cv2

from yo_ratchet.yo_wrangle.common import get_all_txt_recursive, get_all_jpg_recursive


def resize_pad_image(img: numpy.ndarray, scale: float, final_size: int = 640):
    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)
    dsize = (new_width, new_height)
    resized = cv2.resize(img, dsize)

    x_border_pixels = int((final_size - resized.shape[1]) / 2)
    y_border_pixels = int((final_size - resized.shape[0]) / 2)
    padded = cv2.copyMakeBorder(
        resized,
        y_border_pixels,
        y_border_pixels,
        x_border_pixels,
        x_border_pixels,
        cv2.BORDER_CONSTANT,
        value=[127, 127, 127],
    )
    return padded


def resize_pad_all_images_recursively(
    images_root: Path,
    resize_dst: Path,
    final_size: int = 640,
    scale: float = 0.5,
):
    """
    Writes a resized & padded version of all images in dir to a new folder.

    """
    resize_dst.mkdir(exist_ok=True)
    for image_path in get_all_jpg_recursive(img_root=images_root):
        img = cv2.imread(str(image_path))
        padded = resize_pad_image(img=img, scale=scale, final_size=final_size)
        dst_path = resize_dst / image_path.name
        cv2.imwrite(str(dst_path), img=padded)


def un_transform_resized_padded_box(
    box_coords: List[float],
    scale: float,
):
    """
    Reverse transforms a single box that was resized and padded, back to the
    scale and centre appropriate for the original image.

    """
    [x_centre, y_centre, width, height] = box_coords[0:4]
    width = width / scale
    height = height / scale
    x_centre = 0.5 - (0.5 - x_centre) / scale
    y_centre = 0.5 - (0.5 - y_centre) / scale
    return [x_centre, y_centre, width, height]


def un_transform_resize_pad_recursively(
    src_labels_root: Path,
    dst_labels_root: Path,
    scale: float,
):
    """
    Converts bounding box coordinates for a resized-padded transformed image
    back to the original image frame of reference.

    We transform whole txt files, for a whole directory of labels in
    batch mode so all of this data is available prior to calling
    prepare_ai_file_for_virtual_racas().

    """
    dst_labels_root.mkdir(exist_ok=True)
    for annotations_path in get_all_txt_recursive(root_dir=src_labels_root):
        with open(str(annotations_path), "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line_list = line.strip().split(" ")
            class_id = line_list[0]
            box = [float(el) for el in line_list[1:5]]
            un_transformed_box = un_transform_resized_padded_box(
                box_coords=box,
                scale=scale,
            )
            un_transformed_box = [str(el) for el in un_transformed_box]
            new_line = " ".join(un_transformed_box)
            new_line = f"{class_id} " + new_line
            if len(line_list) == 6:
                prob = line_list[5]
                new_line += f" {prob}"
            new_line = (
                new_line + "\n"
            )  # Assume data not directly used on Windows machine
            new_lines.append(new_line)
        dst_path = dst_labels_root / annotations_path.name
        with open(str(dst_path), "w") as f:
            f.writelines(new_lines)


def test_resize_pad():
    """
    Test: Check if two large potholes in Isaac image "Photo_2021_May_02_08_11_50_069_b.jpg"
    are found. Resize and pad all images in: Path("/home/david/RAC/RACAS_Isaac_2021")

    """
    resize_pad_all_images_recursively(
        images_root=Path("/home/david/RACAS/640_x_640/Charters_Towers_subsamples_mix"),
        resize_dst=Path(
            "/home/david/RACAS/640_x_640/Charters_Towers_subsamples_mix_resized_padded"
        ),
    )


def test_un_transform_resize_pad_recursively():
    un_transform_resize_pad_recursively(
        src_labels_root=Path(
            "/home/david/addn_repos/yolov5/runs/detect/Charters_Towers_subsamples_mix_resized_padded__srd19.1_conf10pcnt"
        ),
        dst_labels_root=Path(
            "/home/david/RACAS/640_x_640/Charters_Towers_subsamples_mix_resized_padded/YOLO_darknet"
        ),
        scale=0.5,
    )
