import pandas
from tabulate import tabulate

from pathlib import Path
from typing import List, Tuple, Optional, Dict
from yo_ratchet.yo_wrangle.common import (
    YOLO_ANNOTATIONS_FOLDER_NAME,
    get_all_jpg_recursive,
)


def count_class_instances_in_datasets(
    data_samples: List[Tuple[Path, Optional[int]]],
    class_ids: List[int],
    class_id_to_name_map: Dict[int, str],
):
    """
    Prints a table of instance counts of defect class
    taking the data from YOLO annotation files nested within the
    sample image directory.

    Class names form the columns.
    Dataset names form the rows.

    """
    results_dict = {}
    for dataset_path, max_samples in data_samples:
        dataset_name = dataset_path.stem  # equally, could be dataset_path.name
        annotations_root = dataset_path / YOLO_ANNOTATIONS_FOLDER_NAME
        if not annotations_root.exists():
            annotations_root = dataset_path / "labels"
        if (
            not annotations_root.exists()
        ):  # Annotations may be side-by-side with images.
            annotations_root = dataset_path
        assert annotations_root.exists(), f"{str(annotations_root)} does not exist"

        dataset_dict = {}
        for i, image_path in enumerate(get_all_jpg_recursive(img_root=dataset_path)):
            if i == max_samples:
                print(
                    f"WARNING: Counting stats beyond nominated limit: {dataset_path.name}"
                )
            annotations_file = annotations_root / f"{image_path.stem}.txt"
            if not annotations_file.exists():
                continue
            with open(annotations_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.strip().split(" ")[0])
                    if class_id not in class_ids:
                        continue
                    class_name = class_id_to_name_map.get(class_id, class_id)
                    if not dataset_dict.get(class_name, None):
                        dataset_dict[class_name] = 1
                    else:
                        dataset_dict[class_name] += 1

        results_dict[dataset_name] = dataset_dict

    df = pandas.DataFrame(results_dict).fillna(value=0)
    grand_total = df.sum().sum()
    df["TOTAL"] = df.sum(axis=1)
    df = df[list(df.columns)] = df[list(df.columns)].astype(int)
    df = df.transpose().astype(int)
    unordered_cols = list(df)
    ordered_cols = [
        class_name
        for class_name in class_id_to_name_map.values()
        if class_name in unordered_cols
    ]
    df = df[ordered_cols]
    df["TOTAL"] = df.sum(axis=1)
    output_str = tabulate(
        df,
        headers="keys",
        showindex="always",
        tablefmt="pretty",
        stralign="left",
        numalign="right",
    )
    output_str = output_str + f"\nGrand Total: {grand_total}\n"
    print(output_str)
    return output_str
