from cv2 import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler

from yo_ratchet.yo_wrangle.common import YOLO_ANNOTATIONS_FOLDER_NAME, LABELS_FOLDER_NAME

PATCH_MARGIN = 0.01
PATCH_W = 200
PATCH_H = 200


def get_2048_features_standardiser(path_training_subset: Path, class_id: int):
    train_data_dict_list = get_patches_features_data_dict_list(dataset_root=path_training_subset, class_id=class_id)
    training_df = pd.DataFrame(train_data_dict_list)
    training_features_list = list(training_df["features"])
    training_features_array = np.array(training_features_list, dtype="float64")
    ss = StandardScaler()
    ss_train_features_array = ss.fit_transform(training_features_array)
    return ss, ss_train_features_array, training_df


def get_distance(ss: StandardScaler, image_features):
    ss_row_features_array = ss.transform(image_features.reshape(1, -1))
    rmsd = np.square(ss_row_features_array)
    rmsd = rmsd.mean(axis=1)
    rmsd = np.sqrt(rmsd)
    return rmsd


def get_patches_features_data_dict_list(
    dataset_root: Path,
    class_id: int,
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
                "crop": <np.ndarray of cropped patch>,
                "features": <extracted_features_for_patch>
                "class_id": <class_id>
                "subset": <name of the parent folder to the annotations_dir>,
            },
        ]
        where <patch_id> = f"{image_path.stem}_{<seq patch # for patch in image>}"

    NOTE::
        An alternative function for looping through all the annotations is to use
        wrangle_filtered.filter_detections().

    """
    resnet50 = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    resnet50.layers[0].trainable = False

    intermediate_model = tf.keras.Model(
        inputs=resnet50.input,
        outputs=resnet50.layers[112].output  # layer 80 also good.
    )
    # x = tf.keras.layers.Flatten(name="flatten")(intermediate_model.output)
    # x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(intermediate_model.output)
    o = tf.keras.layers.Activation('sigmoid', name='loss')(x)

    MyModel = tf.keras.Model(inputs=resnet50.input, outputs=[o])
    MyModel.layers[0].trainable = False
    if annotations_dir is None:
        if (dataset_root / LABELS_FOLDER_NAME).exists():
            annotations_dir = dataset_root / LABELS_FOLDER_NAME
        elif (dataset_root / YOLO_ANNOTATIONS_FOLDER_NAME).exists():
            annotations_dir = dataset_root / YOLO_ANNOTATIONS_FOLDER_NAME
        else:
            raise RuntimeError("Please provide an argument for the annotations_dir parameter.")

    annotation_files: List = sorted(annotations_dir.rglob("*.txt"))
    if len(annotation_files) == 0:
        print(f"\nNo files found in {annotations_dir}")

    potential_images_subdir = dataset_root / "images"
    if potential_images_subdir.exists() and potential_images_subdir.is_dir():
        images_root = potential_images_subdir
    else:
        images_root = dataset_root

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
            line_class_id = int(line_split[0])
            if line_class_id not in [class_id]:
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


def _extract_features_for_patch(
    model:  tf.keras.models.Sequential,
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
