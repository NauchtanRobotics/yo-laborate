from pathlib import Path

import fiftyone as fo
import threading
from fiftyone import ViewField, DatasetView
from typing import List, Tuple, Optional

from yo_ratchet.open_labeling_integration.launcher import edit_labels
from yo_ratchet.fiftyone_integration.helpers import print_dataset_info
from yo_ratchet.yo_wrangle.common import inferred_base_dir


def _extract_filenames_by_tag(
    dataset_label: str,
    tag: str = "error",  # Alternatively, can use "eval_fp", "mistakenness" or "eval_fn"
    limit: int = 100,
    processed: bool = True,
    reverse: bool = True,
    label_filters: Optional[List[str]] = None,  # e.g. ['D40', 'PP]
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

    if label_filters:
        dataset = dataset.filter_labels(
            "ground_truth", ViewField("label").is_in(label_filters)
        ).filter_labels("prediction", ViewField("label").is_in(label_filters))
    else:
        pass

    if processed:
        dataset = dataset.match_tags("processed")
    else:
        pass

    if tag.lower() == "mistakenness":
        dataset = dataset.sort_by("mistakenness", reverse=reverse)
        filtered_dataset = dataset.limit(limit)
    elif tag == "error":
        filtered_dataset = dataset.match_tags("error").limit(limit)
    elif tag == "group":
        filtered_dataset = dataset.filter_labels(
            "prediction", ViewField("eval_id") == ""
        ).sort_by(
            ViewField("prediction.detections").map(ViewField("confidence")).max(),
            reverse=True,
        )
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


def find_errors(
    dataset_label: str,
    class_names: List[str],
    tag: str = "eval_fn",
    limit: int = 25,
    processed: bool = True,
    reverse: bool = True,
    label_filters: Optional[List[str]] = None,
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
    else:
        pass
    file_names, filtered_dataset = _extract_filenames_by_tag(
        dataset_label=dataset_label,
        tag=tag,
        limit=limit,
        processed=processed,
        reverse=reverse,
        label_filters=label_filters,
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
        fo.launch_app(filtered_dataset)
    else:
        print("Cannot launch the FiftyOne interface.")
        raise Exception(
            "Your filtered dataset is not a DatasetView. type = {}".format(
                type(filtered_dataset)
            )
        )
