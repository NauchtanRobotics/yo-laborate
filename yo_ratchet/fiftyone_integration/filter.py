from pathlib import Path

import fiftyone as fo
import subprocess
import threading
from fiftyone import ViewField, DatasetView
from typing import List, Tuple, Optional

from open_labeling.launcher import POETRY_APP

from yo_ratchet.fiftyone_integration.helpers import print_dataset_info
from yo_ratchet.yo_wrangle.common import inferred_base_dir


def _extract_filenames_by_tag(
    dataset_label: str,
    tag: str = "error",  # Alternatively, can use "eval_fp", "mistakenness" or "eval_fn"
    limit: int = 100,
    processed: bool = True,
    reverse: bool = True,
    label_filter: Optional[str] = "WS",  # e.g. 'CD'
) -> Tuple[List[str], DatasetView]:
    """Loops through a FiftyOne dataset (corresponding to the dataset_label param) and
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

    if label_filter:
        dataset = dataset.filter_labels(
            "ground_truth", ViewField("label") == label_filter
        )
    else:
        pass

    if processed:
        dataset = dataset.match_tags("processed")
    else:
        pass

    if tag.lower() == "mistakenness":
        dataset = dataset.sort_by("mistakenness", reverse=reverse)
        print("came here", len(dataset))
        filtered_dataset = dataset.limit(limit)
    elif tag == "error":
        filtered_dataset = dataset.match_tags("error").limit(limit)
    else:
        filtered_dataset = dataset
        split_tag = tag.split("_")
        if len(split_tag) == 2 and split_tag[0] == "eval":
            filter_val = split_tag[1]
            if filter_val == "fp":
                filtered_dataset = (
                    filtered_dataset.filter_labels(
                        "prediction", ViewField("eval") == filter_val
                    )
                    .sort_by("uniqueness", reverse=reverse)
                    .limit(limit)
                )
            elif filter_val == "fn":
                filtered_dataset = (
                    filtered_dataset.filter_labels(
                        "ground_truth", ViewField("eval") == filter_val
                    )
                    .sort_by("uniqueness", reverse=reverse)
                    .limit(limit)
                )
            else:
                pass  # Do we really want to examine "tp"?
            filtered_dataset = filtered_dataset.sort_by("filepath")
        else:  # e.g. tag is unknown
            pass

    list_files_to_edit = [x.filepath for x in filtered_dataset]
    return list_files_to_edit, filtered_dataset


def edit_labels(filenames: List[str], class_names: List[str], base_dir: Path):
    """Opens OpenLabeling with this list of images filenames found in root_folder
    as per provided parameters.

    Reduces the effort of manually checking images, identifying possible labelling errors
    then having to manually search for these and edit in another application.

    """
    print("base_dir: ")
    print(str(base_dir))
    cmd = [
        str(POETRY_APP),
        "env",
        "info",
        "--path",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, cwd=str(base_dir))
    open_labeling_app = res.stdout.decode("utf8").splitlines()[0]
    open_labeling_app = Path(open_labeling_app)
    open_labeling_app = (open_labeling_app / "Lib" / "site-packages" / "open_labeling" / "run_app.py").resolve()
    assert open_labeling_app.exists(), f"Path does not exist: {str(open_labeling_app)}"
    print(str(open_labeling_app))
    if len(res.stderr) > 0:
        print(str(res.stderr))
    cmd = [
            str(POETRY_APP),
            "run",
            "python",
            f"{str(open_labeling_app)}",
            "-c",
            *class_names,
            "--files-list",
            *filenames,
        ]
    try:
        res = subprocess.run(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            cwd=str(base_dir)
        )
    except subprocess.CalledProcessError as error:
        print(error)


def find_errors(
    dataset_label: str,
    class_names: List[str],
    tag: str = "eval_fn",
    limit: int = 25,
    processed: bool = True,
    reverse: bool = True,
    label_filter: Optional[str] = "WS",
    base_dir: Path = None,
):
    """Filters a FiftyOne Dataset according to the tag and other parameters
    provided, then Simultaneously opens both OpenLabeling and FiftyOne in
    the browser.

    This function is provided so that the machine learning engineer can both
    see the predictions vs ground truths boxes in browser window (FiftyOne)
    whilst editing the ground truths in OpenLabeling.

    """
    if base_dir is None:
        base_dir = inferred_base_dir()
    print("B A S E  D I R: ", str(base_dir))
    file_names, filtered_dataset = _extract_filenames_by_tag(
        dataset_label=dataset_label,
        tag=tag,
        limit=limit,
        processed=processed,
        reverse=reverse,
        label_filter=label_filter,
    )
    file_names = [Path(file_name).resolve() for file_name in file_names]
    file_names = [str(file_name) for file_name in file_names if file_name.exists()]

    open_labeling_thread = threading.Thread(
        target=edit_labels,  # Pointer to function that will launch OpenLabeling.
        name="OpenLabeling",
        args=[file_names, class_names, base_dir],
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
