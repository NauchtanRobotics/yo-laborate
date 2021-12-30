import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from typing import List, Dict, Optional
from pathlib import Path

from yo_wrangle.common import (
    get_all_jpg_recursive,
    YOLO_ANNOTATIONS_FOLDER_NAME,
    LABELS_FOLDER_NAME,
    PASCAL_VOC_FOLDER_NAME,
)
from yo_wrangle.fiftyone_integration.helpers import print_dataset_info

ACCEPTABLE_ANNOTATION_FOLDERS = [
    YOLO_ANNOTATIONS_FOLDER_NAME,
    LABELS_FOLDER_NAME,
]


def _extract_annotation(line: str, label_mapping: Dict[int, str]):
    line_list = line.replace("\n", "").split(" ")
    class_id: str = line_list[0].strip()
    label = label_mapping.get(int(class_id), "Unknown")
    yolo_box = [float(x) for x in line_list[1:5]]
    if len(line_list) == 6:
        confidence = line_list[5]
    else:
        confidence = None
    return label, yolo_box, confidence


def _get_bounding_box(yolo_box: List[float]) -> List[float]:
    width = yolo_box[2]
    height = yolo_box[3]
    centroid_x = yolo_box[0]
    centroid_y = yolo_box[1]

    top_left_x = centroid_x - width / 2
    top_left_y = centroid_y - height / 2
    bounding_box = [top_left_x, top_left_y, width, height]
    return bounding_box


def _get_subset_folders(
    dataset_root: Path = None, images_root: Path = None
) -> List[Path]:
    """Assumes all folders in the dataset_root are subset (subsample) folders with
    the exception of folders reserved for annotations (including "YOLO_darknet",
    "PASCAL_VOC", and "labels".

    However, if dataset_root is None, will treat the images_root as
    though it is a sole subset. Note: Only accepts a Path for one of dataset_root
    or images_root. The other needs to be set to None.

    Returns a list of the qualifying subset Paths.

    :raises: an exception if both dataset_root and images_root are None,
             or if both of the params evaluate to not None.
    :raises: an exception is dataset_root is provided but it is not a valid
             pathlib.Path to an existing directory.
    """
    if dataset_root is None and images_root is not None:
        subset_folders = [images_root]
    elif dataset_root and images_root is None:
        if not dataset_root.exists() or not dataset_root.is_dir():
            raise Exception(
                "The dataset_root provided is not a path to an existing folder."
            )
        else:
            pass
        subset_folders = [
            folder
            for folder in dataset_root.iterdir()
            if folder.is_dir()
            and folder.name
            not in [
                YOLO_ANNOTATIONS_FOLDER_NAME,
                LABELS_FOLDER_NAME,
                PASCAL_VOC_FOLDER_NAME,
            ]
            and folder.name[0] != "."
        ]
    else:
        raise Exception(
            "You need to provide a Path to either one of "
            "dataset_root (which contains subset sample folders) or images_root. "
            "Do not provide both."
        )
    return sorted(subset_folders)


def _get_annotations_root(subset_folder: Path) -> Path:
    """Find an annotations root directory within a subset folder, where
    a sub-folder is found having a name corresponding to one of the string sin
    ACCEPTABLE_ANNOTATION_FOLDERS.

    Give preference to "YOLO_darknet" over "labels".

    If there are no sub folders within the subset_folder, the annotations_root
    is assumed to be the same as the subset_folder (side by side with the
    images).

    """
    folders = [
        folder
        for folder in subset_folder.iterdir()
        if folder.is_dir() and folder.name in ACCEPTABLE_ANNOTATION_FOLDERS
    ]
    if len(folders) > 0:
        ground_truths_root = sorted(folders)[-1]
    else:
        ground_truths_root = subset_folder
    return ground_truths_root


def _evaluate(dataset_label: str = "Collation_7"):
    """Evaluates the predictions in the `predictions` field with respect to the
    labels in the `ground_truth` field

    """
    dataset = fo.load_dataset(name=dataset_label)
    results = dataset.evaluate_detections(
        "prediction",
        gt_field="ground_truth",
        eval_key="eval",
        compute_mAP=True,
    )
    results.print_report()


def delete_fiftyone_dataset(dataset_label: str):
    if dataset_label in fo.list_datasets():
        fo.delete_dataset(name=dataset_label)
    else:
        pass


def init_fifty_one_dataset(
    dataset_label: str,
    classes_map: Dict[int, str],
    val_inferences_root: Optional[Path],
    train_inferences_root: Optional[Path],
    dataset_root: Optional[Path] = None,
    images_root: Optional[Path] = None,
    ground_truths_root: Optional[Path] = None,
    candidate_subset: Path = None,
    export_to_json: bool = True,
):
    """Returns a fiftyOne dataset with uniqueness, mistakenness and evaluations."""

    subset_folders = _get_subset_folders(dataset_root, images_root)
    samples = []
    for subset_folder in subset_folders:
        if dataset_root is not None and ground_truths_root is None:
            ground_truths_folder = _get_annotations_root(subset_folder=subset_folder)
        else:
            ground_truths_folder = ground_truths_root
        subset_image_paths = list(get_all_jpg_recursive(img_root=subset_folder))
        for image_path in subset_image_paths:
            sample = fo.Sample(filepath=str(image_path))
            detections = []
            ground_truths_path = ground_truths_folder / f"{image_path.stem}.txt"
            if not ground_truths_path.exists():
                print(f"Ground truth not exist: {str(ground_truths_path)}")
                pass  # no detection(s) will be added to this sample.
            else:
                with open(str(ground_truths_path), "r") as file_obj:
                    annotation_lines = file_obj.readlines()

                for line in annotation_lines:
                    label, yolo_box, _ = _extract_annotation(
                        line=line, label_mapping=classes_map
                    )
                    bounding_box = _get_bounding_box(yolo_box=yolo_box)
                    detections.append(
                        fo.Detection(label=label, bounding_box=bounding_box)
                    )
            predictions = []
            inferences_path = None
            if (
                val_inferences_root
                and (val_inferences_root / f"{image_path.stem}.txt").exists()
            ):
                inferences_path = val_inferences_root / f"{image_path.stem}.txt"
                sample.tags.append("val")
            elif (
                train_inferences_root
                and (train_inferences_root / f"{image_path.stem}.txt").exists()
            ):
                inferences_path = train_inferences_root / f"{image_path.stem}.txt"
                sample.tags.append("train")
            else:
                pass  # No 'predictions' to populate

            if inferences_path and inferences_path.exists():
                with open(str(inferences_path), "r") as file_obj:
                    annotation_lines = file_obj.readlines()
                for line in annotation_lines:
                    label, yolo_box, confidence = _extract_annotation(
                        line=line, label_mapping=classes_map
                    )
                    bounding_box = _get_bounding_box(yolo_box=yolo_box)
                    predictions.append(
                        fo.Detection(
                            label=label,
                            bounding_box=bounding_box,
                            confidence=confidence,
                        )
                    )
                sample.tags.append("processed")

            # Store detections in a field name of your choice
            sample["ground_truth"] = fo.Detections(detections=detections)
            sample["prediction"] = fo.Detections(
                detections=predictions
            )  # Should we do this if predictions is empty?
            sample["subset"] = subset_folder.name

            if candidate_subset and subset_folder.name == candidate_subset.name:
                sample.tags.append("candidate")
            else:
                pass
            samples.append(sample)

    # Create dataset
    dataset = fo.Dataset(dataset_label)
    dataset.add_samples(samples)
    dataset.save()
    model = foz.load_zoo_model("wide-resnet101-2-imagenet-torch")
    embeddings = dataset.compute_embeddings(model=model)
    fob.compute_uniqueness(dataset, embeddings=embeddings)
    fob.compute_mistakenness(
        samples=dataset,
        pred_field="prediction",
        label_field="ground_truth",
    )
    _evaluate(dataset_label=dataset_label)
    dataset.persistent = True
    dataset.save()
    if export_to_json:
        dataset.export(
            export_dir="./.export",
            dataset_type=fo.types.FiftyOneDataset,
            export_media=False,
        )


def start(dataset_label: Optional[str] = None):
    """
    You must first run init_fifty_one_dataset.
    """
    if len(fo.list_datasets()) == 0:
        raise RuntimeError("No datasets available. First run init_fiftyone_dataset().")
    elif dataset_label is None:
        dataset_label = sorted(fo.list_datasets())[-1]
    else:
        pass

    if dataset_label not in fo.list_datasets():
        raise RuntimeError(f"{dataset_label} not found.")

    dataset = fo.load_dataset(name=dataset_label)
    print_dataset_info(dataset)
    print(dataset.last())
    fo.launch_app(dataset)
