from pathlib import Path
from typing import Iterable


YOLO_ANNOTATIONS_FOLDER_NAME = "YOLO_darknet"


def get_all_jpg_recursive(img_root: Path) -> Iterable[Path]:
    for item in img_root.rglob("*.jpg"):
        yield item


def get_all_txt_recursive(root_dir: Path) -> Iterable[Path]:
    for item in root_dir.rglob("*.txt"):
        yield item


def get_corrected_photo_name(photo_name: Path, expected_num_parts: int, sep: str = "_"):
    photo_ext = photo_name.suffix
    photo_split = photo_name.name.split(sep)
    len_photo_split = len(photo_split)
    if len_photo_split > expected_num_parts:
        photo_name = "_".join(photo_split[0 : expected_num_parts])
        photo_name = f"{photo_name}{photo_ext}"
    return photo_name
