from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sb

from pathlib import Path
from typing import List, Dict, Optional
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from yo_ratchet.yo_wrangle.common import YOLO_ANNOTATIONS_FOLDER_NAME, LABELS_FOLDER_NAME

PATCH_MARGIN = 0.01
PATCH_W = 200
PATCH_H = 200


def get_2048_features_standardiser(path_training_subset: Path):
    data_training_subset = get_patches_features_data_dict(images_root=path_training_subset)
    training_df = pd.DataFrame(data_training_subset)
    training_features_list = list(training_df["features"])
    training_features_array = np.array(training_features_list, dtype="float64")
    ss = StandardScaler()
    ss_train_features_array = ss.fit_transform(training_features_array)
    return ss, ss_train_features_array, training_df


def view_outliers(path_training_subset: Path, path_test_data: Path):
    ss, ss_train_features_array, training_df = get_2048_features_standardiser(
        path_training_subset=path_training_subset
    )
    test_features_data_dict = get_patches_features_data_dict(images_root=path_test_data)
    num_results = len(test_features_data_dict)
    row_num = 0

    while True:
        row = test_features_data_dict[row_num]
        image_features = row["features"]
        ss_row_features_array = ss.transform(image_features.reshape(1, -1))
        rmsd = np.square(ss_row_features_array)
        rmsd = rmsd.mean(axis=1)
        rmsd = np.sqrt(rmsd)
        delta = "{:.1f}".format(np.round_(rmsd, 1)[0])
        if rmsd[0] > 2.7:
            label = "Outlier"
        else:
            label = "Normal"
        photo_name = row["image_name"]
        label += f" ({delta}) - {row_num + 1} of {num_results}"
        patch = row["crop"]
        cv2.imshow(label, patch)
        key_pressed = cv2.waitKey(0)
        if key_pressed == 81:
            if row_num > 0:
                row_num -= 1
            else:  # Roll around - go to the end of the list
                row_num = num_results - 1
        elif key_pressed == 27:
            break
        elif key_pressed == 83 or key_pressed == 32:
            if row_num < num_results-1:
                row_num += 1
            else:
                row_num = 0
        else:  # Undefined cmd - do nothing
            pass

        cv2.destroyWindow(label)


def test_find_outliers():
    view_outliers(
        path_training_subset=Path("/home/david/RACAS/640_x_640/Clustering/Charters_Towers_subsample"),
        path_test_data=Path("/home/david/RACAS/640_x_640/Clustering/Leopard_blossoms"),
    )


def get_patches_features_data_dict(
    images_root: Path,
    annotations_dir: Optional[Path] = None,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Feature extractor that takes a labels/ directory as input; i.e. only works on training
    data or images for which inferences have already been made so that patches can be
    compared. This is expected to be more powerful than using an unsupervised technique on
    whole images.

    Iterates through all the annotations files in the labels/
    directory and calculates features from the image in images_root.

    Pads in the x and y directions, then resizes padded patches to PATCH_W x PATCH_H
    pixels to ensure the extracted features have uniform dimensions.

    Returns a List of::
        [
            {
                "patch_ref": <patch_id>,
                "image_name": <image_filename>,
                "features": <extracted_features_for_patch>
                "class_id": <class_id>
            },
        ]
        where <patch_id> = f"{image_path.stem}_{<seq patch # for patch in image>}"

    NOTE::
        An alternative function for looping through all the annotations is to use
        wrangle_filtered.filter_detections().

    """
    MyModel = tf.keras.models.Sequential()
    MyModel.add(
        tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
    )
    MyModel.layers[0].trainable = False

    if annotations_dir is None:
        if (images_root / LABELS_FOLDER_NAME).exists():
            annotations_dir = images_root / LABELS_FOLDER_NAME
        elif (images_root / YOLO_ANNOTATIONS_FOLDER_NAME).exists():
            annotations_dir = images_root / YOLO_ANNOTATIONS_FOLDER_NAME
        else:
            raise RuntimeError("Please provide an argument for the annotations_dir parameter.")

    annotation_files: List = sorted(annotations_dir.rglob("*.txt"))
    if len(annotation_files) == 0:
        print(f"\nNo files found in {annotations_dir}")
    if limit is not None and limit < len(annotation_files):
        annotation_files = annotation_files[: limit]
    # average_patch_sizes = {}  # = get_average_patch_sizes(annotations)
    class_centroids = {}  # {<class_id>: {"v1": <v1>, "v2": <v2>}
    results = []
    for file_path in annotation_files:
        with open(str(file_path), "r") as f:
            lines = f.readlines()
        image_path = images_root / f"{file_path.stem}.jpg"
        if not image_path.exists():
            continue
        lines = set(lines)
        for seq, line in enumerate(lines):
            line_split = line.strip().split(" ")
            patch_ref = f"{image_path.stem}_{seq}"
            class_id = int(line_split[0])
            if class_id not in [17]:
                continue
            # class_name = class_info.get(int(class_id))
            # new_w, new_h = average_patch_sizes.get(class_id)
            x, y, w, h = line_split[1:5]  # Ignores class_id which is line[0] and probability which is line[5]
            _extracted_features, crop = _extract_features_for_patch(
                MyModel, image_path, float(x), float(y), float(w), float(h), PATCH_W, PATCH_H
            )
            results.append({
                "patch_ref": patch_ref,
                "image_name": image_path.name,
                "features": _extracted_features,
                "crop": crop,
                "class_id": int(class_id),
                "subset": annotations_dir.parent.name,
            })
    return results


def test_get_feature_maps_for_patches():
    """ Test on Scenic Rim 2021 to see if the leopard tree are flagged as outliers"""
    features_map = get_patches_features_data_dict(
        images_root=Path("/home/david/RACAS/640_x_640/Scenic_Rim_2021_mined_19.1"),
        annotations_dir=None
    )
    print(features_map)


def _extract_features_for_patch(
    model: tf.keras.models.Sequential,
    path_to_image: Path,
    x: float,
    y: float,
    w: float,
    h: float,
    new_h: float,
    new_w: float,
    show_crops: bool = False
):
    """
    Gets the features for each padded patch.

    Padding extends the dimensions of the patch a little to collect a little more context,
    without extending outside the 0-1 domain.

    """
    img = cv2.imread(str(path_to_image))
    img_h, img_w, channels = img.shape

    x1 = np.clip(int((x - w / 2 - PATCH_MARGIN) * img_w), a_min=0, a_max=img_w)
    x2 = np.clip(int((x + w / 2 + PATCH_MARGIN) * img_w), a_min=0, a_max=img_w)
    y1 = np.clip(int((y - h / 2 - PATCH_MARGIN) * img_h), a_min=0, a_max=img_h)
    y2 = np.clip(int((y + h / 2 + PATCH_MARGIN) * img_h), a_min=0, a_max=img_h)
    crop = img[y1:y2, x1:x2, :]
    crop = cv2.resize(crop, (new_h, new_w))
    if show_crops and path_to_image.name == "Photo_2021_Dec_02_11_44_24_165_b.jpg":
        cv2.imshow("Blah", crop)
        cv2.waitKey(0)
    expanded_crop = np.expand_dims(crop, 0)
    expanded_crop = tf.keras.applications.resnet50.preprocess_input(expanded_crop)
    extractedFeatures = model.predict(expanded_crop)
    extractedFeatures = np.array(extractedFeatures)
    extractedFeatures = extractedFeatures.flatten()
    return extractedFeatures, crop
