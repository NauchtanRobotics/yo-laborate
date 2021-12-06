import pandas
from tabulate import tabulate

from pathlib import Path
from typing import List, Tuple
from yo_wrangle.common import (
    get_id_to_label_map,
    YOLO_ANNOTATIONS_FOLDER_NAME,
    get_all_txt_recursive,
)


def count_class_instances_in_datasets(
    data_samples: List[Tuple[Path, int]],
    class_ids: List[int],
    class_names_path: Path,
):
    """
    Prints a table of instance counts of defect class
    taking the data from YOLO annotation files nested within the
    sample image directory.

    Class names form the columns.
    Dataset names form the rows.

    """
    classes_map = get_id_to_label_map(classes_list_path=class_names_path)
    results_dict = {}
    for sample in data_samples:
        dataset_path = sample[0]
        dataset_name = dataset_path.stem  # equally, could be dataset_path.name
        dataset_annotations = dataset_path / YOLO_ANNOTATIONS_FOLDER_NAME
        if not dataset_annotations.exists():
            dataset_annotations = dataset_path  # Will search recursively anyway
        assert (
            dataset_annotations.exists()
        ), f"{str(dataset_annotations)} does not exist"
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

    df = pandas.DataFrame(results_dict).fillna(value=0)
    df["TOTAL"] = df.sum(axis=1)
    df = df[list(df.columns)] = df[list(df.columns)].astype(int)
    df = df.transpose().astype(int)
    output_str = tabulate(
        df,
        headers="keys",
        showindex="always",
        tablefmt="pretty",
        stralign="left",
        numalign="right",
    )
    print("\n")
    print(output_str)
    return output_str
