from pathlib import Path
from typing import Dict

from yo_ratchet.yo_wrangle.common import get_all_txt_recursive


def recode_and_filter(
    annotations_dir: Path,
    recode_map: Dict[str, str],
):
    """
    Recodes from recode_map key to the corresponding recode_map value.

    Filters out any annotations that are not represented by any of the keys.

    The keys and values of recode_map are represented as strings, e.g.::

        recode_map={"10": "0"}

    """
    for annotations_file in get_all_txt_recursive(root_dir=annotations_dir):
        with open(annotations_file, "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line_list = line.strip().split(" ")
            class_id = line_list[0]
            if class_id not in list(recode_map.keys()):
                continue
            new_class_id = recode_map[class_id]
            line_list[0] = new_class_id
            new_line = " ".join(line_list[0:6])
            new_line = new_line + "\n"
            new_lines.append(new_line)
        with open(annotations_file, "w") as f:
            f.writelines(new_lines)


def test_recode_and_filter():
    recode_and_filter(
        annotations_dir=Path(
            "/home/david/RACAS/boosted/600_x_600/unmasked/Signs_Central_Coast_2021/YOLO_darknet"
        ),
        recode_map={
            "10": "0",  # AP predictions were often Signs. I wanted to model only signs hence 0 class id.
        },
    )
