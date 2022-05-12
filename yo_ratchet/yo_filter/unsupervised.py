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


def visualise_clusters(images_root: Path, limit: Optional[int] = 1000):  # , n_clusters: int = 2):
    data_list = get_feature_maps_for_patches(images_root=images_root, limit=limit)
    df = pd.DataFrame(data_list)
    features_list = list(df["features"])
    feature_vector = np.array(features_list, dtype="float64")

    kmeans = AgglomerativeClustering(compute_distances=True)
    kmeans.fit(feature_vector)
    predictions = kmeans.labels_

    dimReducedDataFrame = pd.DataFrame(feature_vector)
    dimReducedDataFrame = dimReducedDataFrame.rename(columns={0: "V1", 1: "V2", 2: "V3"})
    plt.figure(figsize=(10, 5))

    # Apply colour to markers according to known class membership
    dimReducedDataFrame["Category"] = list(df["class_id"])

    sb.scatterplot(data=dimReducedDataFrame, x="V2", y="V3", hue="Category")
    plt.grid(True)
    plt.show()
    print()


def test_visualise():
    """ Test on Scenic Rim 2021 to see if the leopard tree are flagged as outliers"""
    visualise_clusters(
        images_root=Path("/home/david/RACAS/sealed_roads_dataset/Scenic_Rim_2021_mined_1"),
        #n_clusters=26,
    )
    print()


def visualise_clusters_comparing_subsets(path_training_subset: Path, path_test_data: Path):  # , n_clusters: int = 2):
    data_training_subset = get_feature_maps_for_patches(images_root=path_training_subset)
    training_df = pd.DataFrame(data_training_subset)
    training_features_list = list(training_df["features"])
    training_features_array = np.array(training_features_list, dtype="float64")

    data_test_subset = get_feature_maps_for_patches(images_root=path_test_data)
    test_df = pd.DataFrame(data_test_subset)
    test_features_list = list(test_df["features"])
    test_features_array = np.array(test_features_list, dtype="float64")

    concatenated_df = pd.concat([test_df, training_df], ignore_index=True, sort=False)
    aggregated_feature_arrays = np.concatenate([test_features_array, training_features_array])
    # Standardizing the features
    ss = StandardScaler()
    ss_train_features_array = ss.fit_transform(training_features_array)
    ss_test_features_array = ss.transform(test_features_array)
    ss_train_test_features_array = ss.transform(aggregated_feature_arrays)
    # expanded_df = pd.DataFrame(list(ss_train_test_features_array))
    # new_df = pd.concat([concatenated_df, expanded_df], axis=1)
    distances = np.square(ss_train_test_features_array)
    distances = distances.sum(axis=1)
    distances = np.sqrt(distances)
    concatenated_df["All_Dist"] = list(distances)
    concatenated_df["Outlier"] = concatenated_df.apply(
        lambda x: 1 if x["All_Dist"] > 65 else 0,
        axis=1
    )

    pca = PCA()
    pc_array_train = pca.fit_transform(ss_train_features_array)
    pc_array_test = pca.transform(ss_test_features_array)
    aggregated_pc_arrays = np.concatenate([pc_array_test, pc_array_train])
    principal_df = pd.DataFrame(
        data=aggregated_pc_arrays,
    )
    principal_df["Subset"] = list(concatenated_df["subset"])

    # # Identify cluster associations then compare to known classifications
    # kmeans = AgglomerativeClustering(compute_distances=True)
    # kmeans.fit(distances)
    # principal_df["Cluster"] = list(kmeans.labels_)

    # plt.figure(figsize=(10, 5))
    # sb.scatterplot(data=principal_df, x="0", y="1", hue="Subset")
    # plt.grid(True)
    # plt.show()
    print()


def test_visualise_clusters_comparing_subsets():
    """ Test on Scenic Rim 2021 to see if the leopard tree are flagged as outliers"""

    visualise_clusters_comparing_subsets(
        path_training_subset=Path("/home/david/RACAS/640_x_640/Clustering/Charters_Towers_subsample"),
        path_test_data=Path("/home/david/RACAS/640_x_640/Clustering/Leopard_blossoms"),
    )
    print()


def get_feature_maps_for_patches(
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
        for seq, line in enumerate(lines):
            line_split = line.strip().split(" ")
            patch_ref = f"{image_path.stem}_{seq}"
            class_id = int(line_split[0])
            if class_id not in [17]:
                continue
            # class_name = class_info.get(int(class_id))
            # new_w, new_h = average_patch_sizes.get(class_id)
            x, y, w, h = line_split[1:5]  # Ignores class_id which is line[0] and probability which is line[5]
            _extracted_features = _extract_features_for_patch(
                MyModel, image_path, float(x), float(y), float(w), float(h), PATCH_W, PATCH_H
            )
            results.append({
                "patch_ref": patch_ref,
                "image_name": image_path.name,
                "features": _extracted_features,
                "class_id": int(class_id),
                "subset": annotations_dir.parent.name,
            })
    return results


def test_get_feature_maps_for_patches():
    """ Test on Scenic Rim 2021 to see if the leopard tree are flagged as outliers"""
    features_map = get_feature_maps_for_patches(
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
    if path_to_image.name == "Photo_2021_Dec_02_11_44_24_165_b.jpg":
        cv2.imshow("Blah", crop)
        cv2.waitKey(0)
    crop = np.expand_dims(crop, 0)
    crop = tf.keras.applications.resnet50.preprocess_input(crop)
    extractedFeatures = model.predict(crop)
    extractedFeatures = np.array(extractedFeatures)
    extractedFeatures = extractedFeatures.flatten()
    return extractedFeatures
