import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path
from typing import Optional, List

import cv2
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from yo_ratchet.yo_filter.unsupervised import (
    get_patches_features_data_dict_list,
    get_distance_for_vector,
    get_features_matrix_and_df,
    get_rms_distance_vector_for_matrix,
    find_n_most_distant_outliers_in_batch,
)


def test_find_n_most_distant_outliers_in_batch():
    find_n_most_distant_outliers_in_batch(
        train_data=Path("/home/david/addn_repos/yolov5/datasets/srd26.0.1/train"),
        test_data=Path(
            "/home/david/RACAS/sealed_roads_dataset/Scenic_Rim_2022_mined_1"
        ),
        class_id=4,
        layer_number=80,
        n_outliers=6,
    )


def save_cropped_defect_according_to_outlier_status(
    train_data: Path,
    path_test_data: Path,
    class_id: int,
    dst_root: Path,
    control_limit: float = 2.7,
):
    train_features_matrix, training_df = get_features_matrix_and_df(
        subset_path=train_data, class_id=class_id, layer_number=80
    )
    ss = StandardScaler()
    _ = ss.fit_transform(train_features_matrix)
    test_features_data_dict_list = get_patches_features_data_dict_list(
        dataset_root=path_test_data, class_id=class_id
    )
    train_rmsd = get_rms_distance_vector_for_matrix(
        ss=ss, image_features_matrix=train_features_matrix
    )
    mean = train_rmsd.mean(axis=0)
    stddev = train_rmsd.std(axis=0)

    (dst_root / f"class_{class_id}" / "Outlier").mkdir(parents=True)
    (dst_root / f"class_{class_id}" / "Normal").mkdir()
    for image_features_data_dict in test_features_data_dict_list:
        image_features = image_features_data_dict["features"]
        photo_name = image_features_data_dict["image_name"]
        patch_array = image_features_data_dict["crop"]
        rmsd = get_distance_for_vector(ss=ss, image_features=image_features)
        rmsd = rmsd[0]  # get value from array length 1
        delta = (rmsd - mean) / stddev
        if delta > control_limit:
            label = "Outlier"
        else:
            label = "Normal"
        dst_path = dst_root / f"class_{class_id}" / label / photo_name
        cv2.imwrite(str(dst_path), patch_array)


def save_cropped_defect_according_to_outlier_status_multiple_classes(
    train_data: Path,
    test_data: Path,
    output_dir_parent: Path,
    class_ids: List[int],
    layer_number: Optional[int] = 80,
    control_limit_coefficient: Optional[float] = 1.8,
):
    dst_root = (
        output_dir_parent
        / f"{test_data.name}_layer{layer_number}_thresh{control_limit_coefficient}"
    )

    for class_id in class_ids:
        save_cropped_defect_according_to_outlier_status(
            train_data=train_data,
            path_test_data=test_data,
            dst_root=dst_root,
            class_id=class_id,
            control_limit=1.8,
        )


def test_save_cropped_defect_according_to_outlier_status_multiple_classes():
    save_cropped_defect_according_to_outlier_status_multiple_classes(
        train_data=Path("/home/david/addn_repos/yolov5/datasets/srd25.0.1/train"),
        test_data=Path(
            "/home/david/RACAS/sealed_roads_dataset/Scenic_Rim_2022_mined_1"
        ),
        output_dir_parent=Path("/home/david/RACAS"),
        class_ids=[0, 1, 2, 3, 4, 12, 17],
        layer_number=80,
        control_limit_coefficient=1.8,
    )


def step_through_show_outliers_vs_normal(
    path_training_subset: Path, path_test_data: Path, class_id: int
):
    """
    Allows you to manually view each defect crop (patch) and the result of the inference test
    on whether the patch is an outlier or not. The training images provide imagenet features
    which are then used to define the mean and standard deviation for normalise the features
    of the test data.

    Use the forward and backward arrows to index through the test images.

    """
    features_matrix, training_df = get_features_matrix_and_df(
        subset_path=path_training_subset,
        class_id=class_id,
        layer_number=80,
    )
    ss = StandardScaler()
    ss.fit_transform(features_matrix)
    test_features_data_dict = get_patches_features_data_dict_list(
        dataset_root=path_test_data, class_id=class_id
    )
    num_results = len(test_features_data_dict)
    row_num = 0

    while True:
        row = test_features_data_dict[row_num]
        image_features = row["features"]
        rmsd = get_distance_for_vector(ss=ss, image_features=image_features)
        delta = "{:.1f}".format(np.round_(rmsd, 1)[0])
        if rmsd[0] > 3.1:
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
            if row_num < num_results - 1:
                row_num += 1
            else:
                row_num = 0
        else:  # Undefined cmd - do nothing
            pass

        cv2.destroyWindow(label)


def test_find_outliers():
    step_through_show_outliers_vs_normal(
        path_training_subset=Path(
            "/home/david/RACAS/640_x_640/Clustering/Charters_Towers_subsample"
        ),
        path_test_data=Path("/home/david/RACAS/640_x_640/Clustering/Leopard_blossoms"),
        class_id=12,
    )


def compare_ss_feature_distances_by_subset(
    path_training_subset: Path, path_test_data: Path, class_id: int
):
    data_training_subset = get_patches_features_data_dict_list(
        dataset_root=path_training_subset, class_id=class_id
    )
    training_df = pd.DataFrame(data_training_subset)
    training_features_list = list(training_df["features"])
    training_features_array = np.array(training_features_list, dtype="float64")

    data_test_subset = get_patches_features_data_dict_list(
        dataset_root=path_test_data, class_id=class_id
    )
    test_df = pd.DataFrame(data_test_subset)
    test_features_list = list(test_df["features"])
    test_features_array = np.array(test_features_list, dtype="float64")

    concatenated_df = pd.concat([test_df, training_df], ignore_index=True, sort=False)
    aggregated_feature_arrays = np.concatenate(
        [test_features_array, training_features_array]
    )
    # Standardizing the features
    ss = StandardScaler()
    ss_train_features_array = ss.fit_transform(training_features_array)
    ss_test_features_array = ss.transform(test_features_array)
    ss_train_test_features_array = ss.transform(aggregated_feature_arrays)
    # expanded_df = pd.DataFrame(list(ss_train_test_features_array))
    # new_df = pd.concat([concatenated_df, expanded_df], axis=1)
    rmsd = np.square(ss_train_test_features_array)
    rmsd = rmsd.mean(axis=1)
    rmsd = np.sqrt(rmsd)
    concatenated_df["RMS_Dist"] = list(rmsd)
    concatenated_df["Outlier"] = concatenated_df.apply(
        lambda x: True if x["RMS_Dist"] > 2.5 else False, axis=1
    )

    # # Identify cluster associations then compare to known classifications
    # kmeans = AgglomerativeClustering(compute_distances=True)
    # kmeans.fit(distances)
    # principal_df["Cluster"] = list(kmeans.labels_)

    plt.figure(figsize=(10, 5))
    sb.scatterplot(data=concatenated_df, x="0", y="1", hue="Subset")
    plt.grid(True)
    plt.show()
    print()


def test_compare_ss_feature_distances_by_subset():
    """Test on Scenic Rim 2021 to see if the leopard tree are flagged as outliers"""
    compare_ss_feature_distances_by_subset(
        path_training_subset=Path(
            "/home/david/RACAS/640_x_640/Clustering/Charters_Towers_subsample"
        ),
        path_test_data=Path("/home/david/RACAS/640_x_640/Clustering/Leopard_blossoms"),
        class_id=12,
    )
    print()


def compare_pca_ss_distances_by_subset():
    """
    Using PCA is helpful for viewing but actually seemed to reduce the performance of the
    distance based control chart.

    """
    class_id = 12
    path_training_subset = Path(
        "/home/david/RACAS/640_x_640/Clustering/Charters_Towers_subsample"
    )
    path_test_data = Path("/home/david/RACAS/640_x_640/Clustering/Leopard_blossoms")
    features_matrix, training_df = get_features_matrix_and_df(
        subset_path=path_training_subset, class_id=class_id, layer_number=80
    )
    ss = StandardScaler()
    ss_train_features_matrix = ss.fit_transform(features_matrix)

    pca = PCA()
    pca_train_array = pca.fit_transform(ss_train_features_matrix)

    test_features_data_dict = get_patches_features_data_dict_list(
        dataset_root=path_test_data, class_id=class_id
    )
    test_df = pd.DataFrame(test_features_data_dict)
    test_features_list = list(test_df["features"])
    test_features_array = np.array(test_features_list, dtype="float64")
    ss_test_features_array = ss.transform(test_features_array)
    pca_test_array = pca.transform(ss_test_features_array)

    ensemble_df = pd.concat([test_df, training_df], axis=0)
    ensemble_pca_array = np.concatenate([pca_test_array, pca_train_array], axis=0)
    rmsd = np.square(ensemble_pca_array)
    rmsd = rmsd.mean(axis=1)
    rmsd = np.sqrt(rmsd)
    RMS_DISTANCE_COL_NAME = "Distance"
    ensemble_df[RMS_DISTANCE_COL_NAME] = list(rmsd)
    ensemble_df["Outlier"] = ensemble_df.apply(
        lambda x: True if x[RMS_DISTANCE_COL_NAME] > 2.5 else False, axis=1
    )
    print()


def visualise_clusters(images_root: Path, class_id: int, limit: Optional[int] = 1000):
    data_list = get_patches_features_data_dict_list(
        dataset_root=images_root, limit=limit, class_id=class_id
    )
    df = pd.DataFrame(data_list)
    features_list = list(df["features"])
    feature_vector = np.array(features_list, dtype="float64")

    kmeans = AgglomerativeClustering(compute_distances=True)
    kmeans.fit(feature_vector)
    predictions = kmeans.labels_

    dimReducedDataFrame = pd.DataFrame(feature_vector)
    dimReducedDataFrame = dimReducedDataFrame.rename(
        columns={0: "V1", 1: "V2", 2: "V3"}
    )
    plt.figure(figsize=(10, 5))

    # Apply colour to markers according to known class membership
    dimReducedDataFrame["Category"] = list(df["class_id"])

    sb.scatterplot(data=dimReducedDataFrame, x="V2", y="V3", hue="Category")
    plt.grid(True)
    plt.show()
    print()


def test_visualise():
    """Test on Scenic Rim 2021 to see if the leopard tree are flagged as outliers"""
    visualise_clusters(
        images_root=Path(
            "/home/david/RACAS/sealed_roads_dataset/Scenic_Rim_2021_mined_1"
        ),
        class_id=12,
    )
    print()


def test_get_feature_maps_for_patches():
    """Test on Scenic Rim 2021 to see if the leopard tree are flagged as outliers"""
    features_map = get_patches_features_data_dict_list(
        dataset_root=Path("/home/david/RACAS/640_x_640/Scenic_Rim_2021_mined_19.1"),
        annotations_dir=None,
        class_id=12,
    )
    print(features_map)
