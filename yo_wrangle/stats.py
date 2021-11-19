import pandas
from tabulate import tabulate

from pathlib import Path
from typing import List, Tuple
from yo_wrangle.common import get_id_to_label_map, YOLO_ANNOTATIONS_FOLDER_NAME, get_all_txt_recursive


def count_class_instances_in_datasets(
    data_samples: List[Tuple[Path, int]],
    class_ids: List[int],
    classes_list: Path,
):
    """
    Prints a table of instance counts of defect class
    taking the data from YOLO annotation files nested within the
    sample image directory.

    Class names form the columns.
    Dataset names form the rows.

    """
    classes_map = get_id_to_label_map(class_name_list_path=classes_list)
    results_dict = {}
    for sample in data_samples:
        dataset_path = sample[0]
        dataset_name = dataset_path.stem
        dataset_annotations = dataset_path / YOLO_ANNOTATIONS_FOLDER_NAME
        dataset_dict = {}
        for annotations_file in get_all_txt_recursive(root_dir=dataset_annotations):
            with open(annotations_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.strip().split(" ")[0])
                    if class_id not in class_ids:
                        continue
                    class_name = classes_map.get(class_id, class_id)
                    if not dataset_dict.get(class_name, None):
                        dataset_dict[class_name] = 1
                    else:
                        dataset_dict[class_name] += 1

        results_dict[dataset_name] = dataset_dict
    print("\n")
    # print(json.dumps(results_dict, indent=4))
    print(
        tabulate(
            pandas.DataFrame(results_dict).transpose(),
            headers="keys",
            showindex="always",
            tablefmt="pretty",
        )
    )