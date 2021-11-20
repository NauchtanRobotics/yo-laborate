import threading

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
import subprocess
from fiftyone import ViewField, DatasetView
from typing import List, Dict, Tuple
from pathlib import Path

from yo_wrangle.common import get_all_jpg_recursive, get_id_to_label_map, YOLO_ANNOTATIONS_FOLDER_NAME

# DATA_ROOT = Path(__file__).parents[1] / "tests/test_data"
DATA_ROOT = Path("/home/david/addn_repos/yolov5")


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


def init_fifty_one_dataset(
    dataset_label: str,
    images_root: Path,
    label_mapping: Dict[int, str],
    ground_truths_root: Path = None,
    inferences_root: Path = None,
):
    """
    Returns a fiftyOne dataset.

    """
    if dataset_label in fo.list_datasets():
        fo.delete_dataset(name=dataset_label)
    else:
        pass

    samples = []
    for image_path in get_all_jpg_recursive(img_root=images_root):
        sample = fo.Sample(filepath=str(image_path))
        detections = []
        annotations_path = ground_truths_root / f"{image_path.stem}.txt"
        if not annotations_path.exists():
            pass  # is anything required here?
        else:
            with open(str(annotations_path), "r") as file_obj:
                annotation_lines = file_obj.readlines()

            for line in annotation_lines:
                label, yolo_box, _ = _extract_annotation(line=line, label_mapping=label_mapping)
                bounding_box = _get_bounding_box(yolo_box=yolo_box)
                detections.append(
                    fo.Detection(label=label, bounding_box=bounding_box)
                )
        predictions = []
        inferences_path = inferences_root / f"{image_path.stem}.txt"
        if not inferences_path.exists():
            pass  # is anything required here?
        else:
            with open(str(inferences_path), "r") as file_obj:
                annotation_lines = file_obj.readlines()
            for line in annotation_lines:
                label, yolo_box, confidence = _extract_annotation(line=line, label_mapping=label_mapping)
                bounding_box = _get_bounding_box(yolo_box=yolo_box)
                predictions.append(
                    fo.Detection(label=label, bounding_box=bounding_box, confidence=confidence)
                )

        # Store detections in a field name of your choice
        sample["ground_truth"] = fo.Detections(detections=detections)
        sample["prediction"] = fo.Detections(detections=predictions)
        samples.append(sample)

    # Create dataset
    dataset = fo.Dataset(dataset_label)
    dataset.add_samples(samples)

    model = foz.load_zoo_model("wide-resnet101-2-imagenet-torch")
    embeddings = dataset.compute_embeddings(model=model)

    # fob.compute_similarity(dataset)
    # Index by uniqueness using pre-computed embeddings
    fob.compute_uniqueness(dataset, embeddings=embeddings)
    fob.compute_mistakenness(
        samples=dataset,
        pred_field="prediction",
        label_field="ground_truth",
    )
    evaluate(dataset_label=dataset_label)
    dataset.persistent = True
    dataset.save()


def test_init_ds(dataset_label: str = "Collation_7"):
    label_mapping = get_id_to_label_map(class_name_list_path=Path(
        # "C:\\Users\\61419\\OpenLabeling\\main\\class_list.txt"
        "/home/david/Desktop/class_list.txt"
    ))
    if dataset_label in fo.list_datasets():
        fo.delete_dataset(name=dataset_label)
    else:
        pass
    init_fifty_one_dataset(
        dataset_label=dataset_label,
        label_mapping=label_mapping,
        images_root=(
                DATA_ROOT / "datasets/bbox_collation_7_split/train/images"
        ),
        ground_truths_root=(
                DATA_ROOT / "datasets/bbox_collation_7_split/train/labels"
        ),
        inferences_root=(
                DATA_ROOT / "runs/detect/Coll_7_train_Collation_7_scale40pcnt_10conf/labels"
        )
    )


def clean_start(dataset_label: str = "Collation_7"):
    if dataset_label in fo.list_datasets():
        test_init_ds()
    else:
        pass
    start()


def start(dataset_label: str = "Collation_7"):
    if dataset_label not in fo.list_datasets():
        test_init_ds()
    else:
        pass
    dataset = fo.load_dataset(name=dataset_label)
    print_dataset_info(dataset)
    fo.launch_app(dataset)


def print_dataset_info(dataset: DatasetView):
    print(dataset)
    print("\nBrains Runs:")
    print(dataset.list_brain_runs())
    print("Evaluation Runs:")
    print(dataset.list_evaluations())


def evaluate(dataset_label: str = "Collation_7"):
    """
    Evaluate the predictions in the `predictions` field with respect to the
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


def _extract_filenames_by_tag(
    dataset_label: str = "Collation_7",
    tag: str = "error",  # Alternatively, can use "eval_fp", "mistakenness" or "eval_fn"
    limit: int = 100,
    conf_threshold: float = 0.2,
) -> Tuple[Path, List[str], DatasetView]:
    """
    Loops through a FiftyOne dataset (corresponding to the dataset_label param) and
    finds all of the images tagged "error". Alternatively, can filters for the top
    100 samples based on the highest value for "eval_fp" or "eval_fn" "eval_fp_fn" or
    "mistakenness".

    Returns a list of image filenames (without full path), images root folder, and the
    DatasetView corresponding to the listed filenames.

    """
    if dataset_label in fo.list_datasets():
        dataset = fo.load_dataset(name=dataset_label)
    else:
        raise Exception(f"Dataset not found: {dataset_label} ")
    print_dataset_info(dataset)

    # .sort_by(F("predictions.detections").length(), reverse=True)
    dataset = dataset.sort_by("mistakenness", reverse=True)
    if tag == "mistakenness":
        filtered_dataset = dataset.limit(limit)
    elif tag == "error":
        filtered_dataset = dataset.match_tags("error").limit(limit)
    else:
        filtered_dataset = dataset.filter_labels(
            "prediction", ViewField("confidence") > conf_threshold
        )
        split_tag = tag.split("_")
        if len(split_tag) == 2 and split_tag[0] == "eval":
            filter_val = split_tag[1]
            if filter_val == "fp":
                filtered_dataset = filtered_dataset.filter_labels(
                    "prediction", ViewField("eval") == filter_val
                ).limit(limit)
            elif filter_val == "fn":
                filtered_dataset = filtered_dataset.filter_labels(
                    "ground_truth", ViewField("eval") == filter_val
                ).limit(limit)
            else:
                pass  # Do we really want to examine "tp"?
            filtered_dataset = filtered_dataset.sort_by("filepath")
        elif len(split_tag) == 3:
            filtered_dataset = filtered_dataset.filter_labels(
                "ground_truth", ViewField("eval") == "fp" | ViewField("eval") == "fn"
            ).limit(limit)
        else:
            pass

    folder_paths = [Path(x.filepath).parent for x in filtered_dataset]
    consistent_folder = None
    for folder_path in folder_paths:  # Check if multiple folders have been used.
        if consistent_folder is None:
            consistent_folder = folder_path
        elif folder_path != consistent_folder:
            raise Exception("Images come from more than one folder.")
        else:
            pass
    if not (consistent_folder / YOLO_ANNOTATIONS_FOLDER_NAME).exists():
        raise Exception(f"Images folder does not contain sub-folder: {YOLO_ANNOTATIONS_FOLDER_NAME}")
    list_files_to_edit = [x.filename for x in filtered_dataset]
    return consistent_folder, list_files_to_edit, filtered_dataset


def edit_labels(root_folder: Path, filenames: List[str], open_labeling_path: Path):
    """
    Opens OpenLabeling with this list of images filenames found in root_folder
    as per provided parameters.

    Reduces the effort of manually checking images, identifying possible labelling errors
    then having to manually search for these and edit in another application.

    """
    open_labeling_env_python = open_labeling_path / "venv/bin/python"
    open_labeling_script = open_labeling_path / "main/main.py"
    cmd = [
        f"{str(open_labeling_env_python)}",
        f"{str(open_labeling_script)}",
        "-i",
        str(root_folder),
        "-o",
        str(root_folder),
        "-l",
        *filenames,
    ]
    subprocess.run(cmd)


def find_errors(
    dataset_label: str,
    open_labeling_dir: Path,
    tag: str = "eval_fn",
    conf_thresh: float = 0.25,
    limit: int = 25,
):
    """
    Filters a FiftyOne Dataset according to the tag and other parameters
    provided, then Simultaneously opens both OpenLabeling and FiftyOne in
    the browser.

    This function is provided so that the machine learning engineer can both
    see the predictions vs ground truths boxes in browser window (FiftyOne)
    whilst editing the ground truths in OpenLabeling.

    """
    root_folder, filenames, filtered_dataset = _extract_filenames_by_tag(
        dataset_label=dataset_label,
        tag=tag,
        conf_threshold=conf_thresh,
        limit=limit,
    )

    open_labeling_thread = threading.Thread(
        target=edit_labels,  # Pointer to function that will launch OpenLabeling.
        name="Downloader",
        args=[root_folder, filenames, open_labeling_dir],
    )
    open_labeling_thread.start()

    if isinstance(filtered_dataset, DatasetView):
        print_dataset_info(filtered_dataset)
        session = fo.launch_app(filtered_dataset)
    else:
        print("Cannot launch the FiftyOne interface.")
        raise Exception(
            "Your filtered dataset is not a DatasetView. type = {}".format(
                type(filtered_dataset)
            )
        )


def test_find_errors():
    find_errors(
        dataset_label="Collation_7",
        open_labeling_dir=Path("/home/david/addn_repos/OpenLabeling"),
        tag="eval_fn",
        conf_thresh=0.5,
        limit=25,
    )
