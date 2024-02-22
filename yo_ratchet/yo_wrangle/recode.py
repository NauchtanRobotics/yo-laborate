from pathlib import Path
from typing import Dict

from yo_ratchet.yo_wrangle.common import get_all_txt_recursive


def recode_using_class_mapping(
    annotations_dir: Path,
    recode_map: Dict[int, int],
    only_retain_mapped_keys: bool = False,
):
    """
    Recodes class_id values in all annotation files recursively in some
    root directory, according to the recode map provided. For example, if

        recode_map={10: 0}

    then all annotation files in annotations_dir will have lines corresponding
    to a class id of 10 changed to a class id of 0.

    Any class_id not included in recode_map.keys() will be retained unchanged,
    unless the only_retain_mapped_keys param is set to True in which case all other
    class ids will be dropped.

    """
    for annotations_file in get_all_txt_recursive(root_dir=annotations_dir):
        with open(annotations_file, "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line_list = line.strip().split(" ")
            class_id = int(line_list[0])
            if class_id in list(recode_map.keys()):
                new_class_id = recode_map[class_id]
                line_list[0] = str(new_class_id)
                new_line = " ".join(line_list[0:6])
                new_line = new_line + "\n"
            else:
                if only_retain_mapped_keys:
                    continue
                new_line = line
            new_lines.append(new_line)
        with open(annotations_file, "w") as f:
            f.writelines(new_lines)


def test_recode_and_filter():
    """
    Class 10 predictions are often 'Signs'. To process a dataset to filter and retain only
    class 10 recoding to a value of 0 so that a single model class can be trained just to
    detect 'Signs' as follows:

    """
    anno_path = Path.home() / "traffic_signs_dataset/GTSDB_2013_JPG/YOLO_darknet"
    assert anno_path.exists() and len(list(anno_path.iterdir())) > 0
    recode_using_class_mapping(
        annotations_dir=anno_path,
        recode_map={
            13:7,
            14:8
        },
        only_retain_mapped_keys=False,
    )
