import pandas
from tabulate import tabulate

from pathlib import Path
from typing import List, Dict, Optional
from yo_ratchet.yo_wrangle.common import (
    YOLO_ANNOTATIONS_FOLDER_NAME,
    get_all_jpg_recursive,
    get_classes_json_path,
    get_id_to_label_map,
)


def count_images_having_confirmed_or_denied_boxes(yolo_file: Path, class_to_count: Optional[int] = None) -> int:
    """
    Prints number of unique images references in a yolo file, plus
    optionally prints the number of bounding boxes fo a single selected class.

    """
    with open(str(yolo_file), "r") as f:
        lines = f.readlines()

    hit_list = set()  # Identify photos that have had at least one defect confirmed or deleted
    count_selected_class = 0
    for line in lines:
        line_split = line.split(" ")
        conf = float(line_split[6])
        if 0 < conf < 1:  # 2, 1 and 0 are the probabilities for manually added, confirmed and denied annotations resp.
            continue  # Only accept confirmed or denied.
        class_idx = int(line_split[1])
        if class_to_count is not None and class_idx == class_to_count:
            count_selected_class += 1
        photo_name = line_split[0]
        hit_list.add(photo_name)
    num_images = len(hit_list)
    print("\nNumber of audited images: " + str(num_images))
    if class_to_count is not None:
        print("\nNumber of selected class boxes: " + str(count_selected_class))
    return num_images


def count_class_instances_in_datasets(
    data_samples: List[Path],
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
    images_count = 0
    for dataset_path in data_samples:
        dataset_name = dataset_path.name
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
            images_count += 1
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
    output_str += "\nTotal number of images (prior to train/val split): " + str(images_count)
    output_str += f"\nGrand Total boxes: {grand_total}\n"
    print(output_str)
    return output_str


def count_class_instances_in_test_datasets(base_dir: Path):
    test_data = base_dir / ".test_datasets"
    sample_folders = [x for x in test_data.iterdir() if (x.is_dir() and x.name[0] != "." and x.name[0] != "_")]
    classes_json_path = get_classes_json_path(base_dir=base_dir)
    classes_map = get_id_to_label_map(Path(f"{classes_json_path}").resolve())
    class_ids = list(classes_map.keys())
    print()
    output_str = count_class_instances_in_datasets(
        data_samples=sample_folders,
        class_ids=class_ids,
        class_id_to_name_map=classes_map,
    )
    return output_str
