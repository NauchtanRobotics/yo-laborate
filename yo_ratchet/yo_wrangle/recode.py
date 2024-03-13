from pathlib import Path
from typing import Dict, Tuple, List

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


def recode_to_severity(
    annotations_dir: Path,
    recode_map: Dict[int, Tuple[int, float]],
    default_severity: float = 0,
    only_retain_mapped_keys: bool = False,
):
    """
    Recodes data in yolo.txt annotation files to classID, bounding box parameters, plus a fifth regression
    output for risk-level/intensity/severity/depth or other value parameter.  recode_map is a required parameter
    keyed by current classID to a tuple of the new designated classID, and the average 'severity' value for the
    in-coming class contributions. Example recode_map showing aggregation into classes 3, 6 and 12:

    recode_map={
          34: (6, 4.5),
          22: (6, 3.3),
          6:  (6, 2.5),
          17: (12, 4.0),
          33: (12, 4.5),
          20: (12, 3.0),
          24: (12, 3.5),
          18: (3, 2),
          3: (3, 4)
        }

    A class can recode to the same class in order to provide a severity for contributions from that original
    class.

    All unmapped classes will be assigned a severity according to the default_severity parameter.

    Any class_id not included in recode_map.keys() will be retained unchanged,
    unless the only_retain_mapped_keys param is set to True in which case all other
    class ids will be dropped.

    FUTURE: Consider capability to recode to two classes (e.g. rutting cracks, PPF to rutting and stripping)

    """
    for annotations_file in get_all_txt_recursive(root_dir=annotations_dir):
        with open(annotations_file, "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line_list: List[str] = line.strip().split(" ")
            class_id = int(line_list[0])
            if class_id in list(recode_map.keys()):
                new_class_id, severity = recode_map[class_id]
                line_list[0] = str(new_class_id)
                if len(line_list) == 5:
                    line_list.append(str(severity))
                else:
                    line_list[5] = str(severity)
            else:
                if only_retain_mapped_keys:
                    continue
                if len(line_list) == 5:
                    line_list.append(str(default_severity))
                else:
                    line_list[5] = str(default_severity)
            new_line = " ".join(line_list[0:7])
            new_line = new_line + "\n"
            new_lines.append(new_line)
        with open(annotations_file, "w") as f:
            f.writelines(new_lines)

