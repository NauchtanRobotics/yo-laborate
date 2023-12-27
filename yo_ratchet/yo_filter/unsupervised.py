import cv2
import numpy as np
import pandas as pd
# import tensorflow as tf

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler

from yo_ratchet.yo_wrangle.common import (
    YOLO_ANNOTATIONS_FOLDER_NAME,
    LABELS_FOLDER_NAME,
)

PATCH_MARGIN = 0.01
PATCH_W = 200
PATCH_H = 200

FEATURES_STR = "features"
IMAGE_NAME_STR = "image_name"
SUBSET_STR = "subset"
DELTA_STR = "delta"


class OutlierDetectionConfig:
    def __init__(
        self,
        training_data_path: Path,
        control_limits_coefficient: float,
        feature_layer_number: int,
    ):
        self.training_data_path = training_data_path
        self.control_limit_coefficient = control_limits_coefficient
        self.feature_layer_number = feature_layer_number


class FeatureNormalisingParams:
    def __init__(self, ss, mean, stddev):
        self.ss: StandardScaler = ss
        self.mean: float = mean
        self.stddev: float = stddev


class OutlierParams:
    def __init__(
        self, outlier_config: OutlierDetectionConfig, classes_info: Dict[str, Dict]
    ):
        self.feature_extraction_model = get_feature_extraction_model(
            layer_number=outlier_config.feature_layer_number
        )
        self.outlier_config: OutlierDetectionConfig = outlier_config
        self.normalising_params: Dict[
            int, FeatureNormalisingParams
        ] = get_all_outlier_normalising_parameters(
            classes_info=classes_info, config=outlier_config
        )


def find_n_most_distant_outliers_in_batch(
    train_data: Path,
    test_data: Path,
    class_id: int,
    layer_number: int,
    n_outliers: int = 5,
) -> List[str]:
    """
    Returns a list of image names in a given subset, test_data, for the least central
    patches for a given class. This is achieved by normalising features found test_data
    based on the sample of features provided by train_data. Features are extracted from
    the output of layer <layer_number> for a resnet50 model trained for 1000 classes.

    The length of the returned list will be n_outliers unless there was a very small
    set of patches found in the test_data subset associated with class_id, in which
    case the length of returned list will be at most a third of the number of images containing
    class=<class_id>.

    Only processes one class at a time.

    Ideally it would process all classes at same time,
    but at this stage get_patches_features_data_dict_list()
    only does one class at a time.

    """
    ss, mean, stddev = get_outlier_normalising_parameters(
        train_data=train_data, class_id=class_id, layer_number=layer_number
    )
    test_features_matrix, test_df = get_features_matrix_and_df(
        subset_path=test_data, class_id=class_id, layer_number=layer_number
    )
    rmsd = get_rms_distance_vector_for_matrix(
        ss=ss, image_features_matrix=test_features_matrix
    )
    delta = (rmsd - mean) / stddev
    test_df[DELTA_STR] = delta
    # Now get the image names for the n most distant patches
    test_df = test_df.sort_values(by=[DELTA_STR], ascending=False)
    image_names = list(set(test_df[IMAGE_NAME_STR]))
    len_results = len(test_df)
    if len_results == 0:
        return []
    elif len_results < n_outliers * 2:
        n_outliers = max(int(round(len_results / 2, 0)), 1)
    else:  # Just return the first n_outliers results as requested
        pass
    image_names = image_names[:n_outliers]
    return image_names


def get_outlier_normalising_parameters(
    train_data: Path,
    class_id: int,
    layer_number: int = 80,
) -> Optional[FeatureNormalisingParams]:
    train_features_matrix, training_df = get_features_matrix_and_df(
        subset_path=train_data, class_id=class_id, layer_number=layer_number
    )

    if train_features_matrix is None or len(training_df) < 100:
        raise RuntimeError(
            f"Insufficient data for class_id {class_id} to generate normalising parameters."
        )

    ss = StandardScaler()
    _ = ss.fit_transform(train_features_matrix)

    train_rmsd = get_rms_distance_vector_for_matrix(
        ss=ss, image_features_matrix=train_features_matrix
    )
    mean = train_rmsd.mean(axis=0)
    stddev = train_rmsd.std(axis=0)
    return FeatureNormalisingParams(ss=ss, mean=mean, stddev=stddev)


def get_all_outlier_normalising_parameters(
    classes_info: Dict[str, Dict],
    config: OutlierDetectionConfig,
) -> Dict[int, FeatureNormalisingParams]:
    """
    Do I want to have different layer numbers for each class?

    NEED TO MAKE ROBUST - what if there is no data for one class?

    Returns a dict conforming to:
    {
        <class_id>: {
            "ss": standard_scalar,
            "mean": mean,
            "stddev": stddev,
        },
        ...
    }
    """
    normalising_parameters_dict = {}
    for class_id in classes_info.keys():
        try:
            feature_normaliser = get_outlier_normalising_parameters(
                train_data=config.training_data_path,
                class_id=int(class_id),
                layer_number=config.feature_layer_number,
            )
        except:
            continue

        if not feature_normaliser.stddev or feature_normaliser.stddev == 0.0:
            continue
        normalising_parameters_dict[int(class_id)] = feature_normaliser
    return normalising_parameters_dict


def get_features_matrix_and_df(
    subset_path: Path, class_id: int, layer_number: int
) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    A dataframe is required when later when processing test data, and we wish
    to match the photo name to an outlier set of features.

    """
    data_dict_list = get_patches_features_data_dict_list(
        dataset_root=subset_path, class_id=class_id, layer_number=layer_number
    )
    df = pd.DataFrame(data_dict_list)
    if len(df) == 0:
        return None, None
    else:
        features_list = list(df[FEATURES_STR])
        features_matrix = np.array(features_list, dtype="float64")
        return features_matrix, df


def get_distance_for_vector(ss: StandardScaler, image_features: np.ndarray):
    ss_row_features_array = ss.transform(image_features.reshape(1, -1))
    rmsd = np.square(ss_row_features_array)
    rmsd = rmsd.mean(axis=1)
    rmsd = np.sqrt(rmsd)
    return rmsd


def get_delta_for_patch(
    class_id: int,
    yolo_box: List[float],
    image_path: Path,
    outlier_params: OutlierParams,
) -> float:
    x, y, w, h = yolo_box
    patch_features, _ = _extract_features_for_patch(
        model=outlier_params.feature_extraction_model,
        path_to_image=image_path,
        x=float(x),
        y=float(y),
        w=float(w),
        h=float(h),
        new_h=PATCH_H,
        new_w=PATCH_W,
    )
    ss = outlier_params.normalising_params[int(class_id)].ss
    return get_distance_for_vector(ss=ss, image_features=patch_features)[0]


def get_rms_distance_vector_for_matrix(
    ss: StandardScaler, image_features_matrix: np.ndarray
):
    standardised_features_matrix = ss.transform(image_features_matrix)
    rmsd = np.square(standardised_features_matrix)
    rmsd = rmsd.mean(axis=1)
    rmsd = np.sqrt(rmsd)
    return rmsd


# def get_feature_extraction_model(layer_number: int = 80) -> "tf.keras.Model":
#     import tensorflow as tf
#     resnet50 = tf.keras.applications.ResNet50(
#         include_top=False,
#         weights="imagenet",
#         pooling="avg",
#     )
#     resnet50.layers[0].trainable = False
#
#     intermediate_model = tf.keras.Model(
#         inputs=resnet50.input,
#         outputs=resnet50.layers[layer_number].output,  # layer 80 also good.
#     )
#     # x = tf.keras.layers.Flatten(name="flatten")(intermediate_model.output)
#     # x = tf.keras.layers.Dense(512, activation='relu')(x)
#     x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(intermediate_model.output)
#     o = tf.keras.layers.Activation("sigmoid", name="loss")(x)
#
#     feature_extraction_model = tf.keras.Model(inputs=resnet50.input, outputs=[o])
#     feature_extraction_model.layers[0].trainable = False
#     return feature_extraction_model
#
#
# def get_patches_features_data_dict_list(
#     dataset_root: Path,
#     class_id: int,
#     annotations_dir: Optional[Path] = None,
#     limit: Optional[int] = None,
#     layer_number: int = 50,  # Tested on: 80 | 112
# ) -> List[Dict]:
#     """
#     Feature extractor that takes a labels/ directory as input; i.e. only works on training
#     data or images for which inferences have already been made so that patches can be
#     compared. This is expected to be more powerful than using an unsupervised technique on
#     whole images.
#
#     Iterates through all the annotations files in the labels/
#     directory and calculates features from the image in images_root.
#
#     Pads in the x and y directions, then resizes padded patches to PATCH_W x PATCH_H
#     pixels to ensure the extracted features have uniform dimensions.
#
#     Returns a List of::
#         [
#             {
#                 "patch_ref": <patch_id>,
#                 "image_name": <image_filename>,
#                 "crop": <np.ndarray of cropped patch>,
#                 "features": <extracted_features_for_patch>
#                 "class_id": <class_id>
#                 "subset": <name of the parent folder to the annotations_dir>,
#             },
#         ]
#         where <patch_id> = f"{image_path.stem}_{<seq patch # for patch in image>}"
#
#     NOTE::
#         An alternative function for looping through all the annotations is to use
#         wrangle_filtered.filter_detections().
#
#     """
#     MyModel = get_feature_extraction_model(layer_number)
#     if (
#         dataset_root.name == YOLO_ANNOTATIONS_FOLDER_NAME
#         and (dataset_root.parent / YOLO_ANNOTATIONS_FOLDER_NAME).exists()
#     ):  # For when accidental double-click down to the next folder.
#         dataset_root = dataset_root.parent
#     else:
#         pass
#
#     if annotations_dir is None:
#         if (dataset_root / LABELS_FOLDER_NAME).exists():
#             annotations_dir = dataset_root / LABELS_FOLDER_NAME
#         elif (dataset_root / YOLO_ANNOTATIONS_FOLDER_NAME).exists():
#             annotations_dir = dataset_root / YOLO_ANNOTATIONS_FOLDER_NAME
#         else:
#             raise RuntimeError(
#                 "Please provide an argument for the annotations_dir parameter."
#             )
#
#     annotation_files: List = sorted(annotations_dir.rglob("*.txt"))
#     if len(annotation_files) == 0:
#         print(f"\nNo files found in {annotations_dir}")
#
#     potential_images_subdir = dataset_root / "images"
#     if potential_images_subdir.exists() and potential_images_subdir.is_dir():
#         images_root = potential_images_subdir
#     else:
#         images_root = dataset_root
#
#     if limit is not None and limit < len(annotation_files):
#         annotation_files = annotation_files[:limit]
#     # average_patch_sizes = {}  # = get_average_patch_sizes(annotations)
#     class_centroids = {}  # {<class_id>: {"v1": <v1>, "v2": <v2>}
#     results = []
#     for file_path in annotation_files:
#         with open(str(file_path), "r") as f:
#             lines = f.readlines()
#         image_path = images_root / f"{file_path.stem}.jpg"
#         if not image_path.exists():
#             continue
#         lines = set(lines)
#         for seq, line in enumerate(lines):
#             line_split = line.strip().split(" ")
#             patch_ref = f"{image_path.stem}_{seq}"
#             line_class_id = int(line_split[0])
#             if line_class_id not in [class_id]:
#                 continue
#             # class_name = class_info.get(int(class_id))
#             # new_w, new_h = average_patch_sizes.get(class_id)
#             x, y, w, h = line_split[
#                 1:5
#             ]  # Ignores class_id which is line[0] and probability which is line[5]
#             _extracted_features, crop = _extract_features_for_patch(
#                 MyModel,
#                 image_path,
#                 float(x),
#                 float(y),
#                 float(w),
#                 float(h),
#                 PATCH_W,
#                 PATCH_H,
#             )
#             results.append(
#                 {
#                     "patch_ref": patch_ref,
#                     IMAGE_NAME_STR: image_path.name,
#                     FEATURES_STR: _extracted_features,
#                     "crop": crop,
#                     "class_id": int(class_id),
#                     "subset": annotations_dir.parent.name,
#                 }
#             )
#     return results
#
#
# def _extract_features_for_patch(
#     model: "tf.keras.models.Sequential",
#     path_to_image: Path,
#     x: float,
#     y: float,
#     w: float,
#     h: float,
#     new_h: float,
#     new_w: float,
#     show_crops: bool = False,
# ):
#     """
#     Gets the features for each padded patch.
#
#     Padding extends the dimensions of the patch a little to collect a little more context,
#     without extending outside the 0-1 domain.
#
#     """
#     import tensorflow as tf
#     img = cv2.imread(str(path_to_image))
#     img_h, img_w, channels = img.shape
#
#     x1 = np.clip(int((x - w / 2 - PATCH_MARGIN) * img_w), a_min=0, a_max=img_w)
#     x2 = np.clip(int((x + w / 2 + PATCH_MARGIN) * img_w), a_min=0, a_max=img_w)
#     y1 = np.clip(int((y - h / 2 - PATCH_MARGIN) * img_h), a_min=0, a_max=img_h)
#     y2 = np.clip(int((y + h / 2 + PATCH_MARGIN) * img_h), a_min=0, a_max=img_h)
#     crop = img[y1:y2, x1:x2, :]
#     crop = cv2.resize(crop, (new_h, new_w))
#     if show_crops and path_to_image.name == "Photo_2021_Dec_02_11_44_24_165_b.jpg":
#         cv2.imshow("Blah", crop)
#         cv2.waitKey(0)
#     expanded_crop = np.expand_dims(crop, 0)
#     expanded_crop = tf.keras.applications.resnet50.preprocess_input(expanded_crop)
#     extractedFeatures = model.predict(expanded_crop)
#     extractedFeatures = np.array(extractedFeatures)
#     extractedFeatures = extractedFeatures.flatten()
#     return extractedFeatures, crop
