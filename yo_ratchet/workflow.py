from pathlib import Path
from typing import Optional, List

from open_labeling import launcher

from yo_ratchet.dataset_versioning import commit_and_push
from yo_ratchet.fiftyone_integration import (
    init_fifty_one_dataset,
    delete_fiftyone_dataset,
    find_errors,
)
from yo_ratchet.fiftyone_integration.create import (
    init_fifty_one_dataset_for_cross_validation_combinations,
)
from yo_ratchet.dataset_versioning.version import (
    bump_patch,
    get_dataset_label_from_version,
)
from yo_ratchet.yo_wrangle.common import (
    get_config_items,
    get_id_to_label_map,
    inferred_base_dir,
    get_classes_list,
    get_subsets_included,
)
from yo_ratchet.modelling import (
    run_detections,
    prepare_dataset_and_train,
)
from yo_ratchet.yo_valuate.as_classification import (
    optimise_binary_and_get_group_classification_performance,
    classification_metrics_for_cross_validation_set,
)

K_FOLDS = 6

YOLO_ROOT = Path()
DATASET_ROOT = Path()
CLASSES_JSON_PATH = Path()
CLASSES_MAP = {}
DST_ROOT = Path()
CONF = 0.05
CONF_PCNT = int(CONF * 100)
TEST_SET_LABEL = ""
BASE_DIR = Path()
GROUPINGS = {}
RECODE_MAP = {}

SUBSETS_INCLUDED = []
EVERY_NTH_TO_VAL = 0
KEEP_CLASS_IDS = []
SKIP_CLASS_IDS = []
DATASET_LABEL = ""


def set_globals(base_dir: Path, workbook_ptr):
    global YOLO_ROOT, DATASET_ROOT, CLASSES_JSON_PATH, CLASSES_MAP
    global DST_ROOT, CONF_PCNT
    global BASE_DIR, CONF, GROUPINGS, SUBSETS_INCLUDED, EVERY_NTH_TO_VAL
    global KEEP_CLASS_IDS, SKIP_CLASS_IDS, RECODE_MAP, DATASET_LABEL

    CONF = workbook_ptr.CONF
    SUBSETS_INCLUDED = get_subsets_included(base_dir)
    EVERY_NTH_TO_VAL = workbook_ptr.EVERY_NTH_TO_VAL
    KEEP_CLASS_IDS = workbook_ptr.KEEP_CLASS_IDS
    SKIP_CLASS_IDS = workbook_ptr.SKIP_CLASS_IDS
    DATASET_LABEL = workbook_ptr.DATASET_LABEL
    RECODE_MAP = workbook_ptr.RECODE_MAP

    GROUPINGS = workbook_ptr.GROUPINGS
    BASE_DIR = base_dir
    _, yolo_root, _, _, _, dataset_root, classes_json_path = get_config_items(
        base_dir=base_dir
    )
    YOLO_ROOT = Path(yolo_root)
    DATASET_ROOT = Path(dataset_root)
    CLASSES_JSON_PATH = Path(classes_json_path)
    CLASSES_MAP = get_id_to_label_map(CLASSES_JSON_PATH)
    DST_ROOT = Path(YOLO_ROOT) / f"datasets/{workbook_ptr.DATASET_LABEL}"
    CONF_PCNT = int(workbook_ptr.CONF * 100)


def launch_open_labeling_folder_browser():
    class_names_list = get_classes_list(base_dir=inferred_base_dir())

    class Args:
        class_list = class_names_list

    args = Args()
    launcher.main(args=args)


def get_labels_and_paths_tuple(dataset_label: str, reverse_it: bool = False):
    if reverse_it:
        test_set_str = "train"
        model_label = f"{dataset_label}_reverse"

    else:
        test_set_str = "val"
        model_label = dataset_label
    test_set_part_label = f"{dataset_label}_{test_set_str}"
    ground_truth_path = DST_ROOT / test_set_str / "labels"
    run_name = f"{test_set_part_label}__{model_label}_conf{CONF_PCNT}pcnt"
    inferences_path = YOLO_ROOT / f"runs/detect/{run_name}/labels"
    return model_label, test_set_part_label, ground_truth_path, inferences_path


def run_prepare_dataset_and_train(
    run_training=True,
    init_fiftyone=True,
    cross_validation_index: int = 0,
    every_n_th: Optional[int] = None,
    fine_tune_epochs: Optional[int] = 5,
    epochs: Optional[int] = 300,
):
    if every_n_th is None:
        every_n_th = EVERY_NTH_TO_VAL
    else:
        pass  # Use the new parameter
    prepare_dataset_and_train(
        classes_map=CLASSES_MAP,
        subsets_included=SUBSETS_INCLUDED,
        dst_root=DST_ROOT,
        every_n_th=every_n_th,
        keep_class_ids=KEEP_CLASS_IDS,
        skip_class_ids=SKIP_CLASS_IDS,
        base_dir=BASE_DIR,
        run_training=run_training,
        recode_map=RECODE_MAP,
        cross_validation_index=cross_validation_index,
        fine_tune_patience=fine_tune_epochs,
        epochs=epochs,
    )
    detect_images_root = DST_ROOT / "val" / "images"
    (
        model_label,
        test_set_part_label,
        ground_truth_path,
        inferences_path,
    ) = get_labels_and_paths_tuple(dataset_label=DATASET_LABEL, reverse_it=False)
    run_detections(
        images_path=detect_images_root,
        dataset_version=test_set_part_label,
        model_path=Path(f"{YOLO_ROOT}/runs/train/{model_label}/weights/best.pt"),
        model_version=model_label,
        base_dir=BASE_DIR,
        conf_thres=CONF,
        device=0,
    )
    optimise_binary_and_get_group_classification_performance(
        images_root=detect_images_root,
        root_ground_truths=ground_truth_path,
        root_inferred_bounding_boxes=inferences_path,
        classes_map=CLASSES_MAP,
        print_first_n=24,
        groupings=GROUPINGS,
        base_dir=BASE_DIR,
    )

    delete_fiftyone_dataset(dataset_label=DATASET_LABEL)
    if init_fiftyone:
        init_fifty_one_dataset(
            dataset_label=DATASET_LABEL,
            classes_map=CLASSES_MAP,
            train_inferences_root=None,
            val_inferences_root=inferences_path,
            dataset_root=DATASET_ROOT,
            images_root=None,  # Use dataset_root approach
            ground_truths_root=None,  # Use dataset_root approach
            export_to_json=True,
        )
    commit_and_push(
        dataset_label=DATASET_LABEL,
        base_dir=BASE_DIR,
        description="Post-training artifacts.",
    )


def cross_validation_combinations_training(
    base_dir: Path,
    init_fiftyone: bool = True,
    k_folds: Optional[int] = K_FOLDS,
    fine_tune_epochs: Optional[int] = 5,
    epochs: Optional[int] = 300,
    training_imgsz: Optional[int] = 640,
    batch_size: Optional[int] = 62,
    cache: Optional[str] = "ram",
    test_imgsz: Optional[int] = 800,
):
    fiftyone_dataset_label = get_dataset_label_from_version(base_dir=base_dir)
    val_inferences_roots = []
    for cv_index in range(k_folds):
        bump_patch(base_dir=base_dir)
        dataset_label = get_dataset_label_from_version(base_dir=base_dir)
        dst_root = Path(YOLO_ROOT) / f"datasets/{dataset_label}"

        prepare_dataset_and_train(
            classes_map=CLASSES_MAP,
            subsets_included=SUBSETS_INCLUDED,
            dst_root=dst_root,
            every_n_th=k_folds,
            keep_class_ids=KEEP_CLASS_IDS,
            skip_class_ids=SKIP_CLASS_IDS,
            base_dir=base_dir,
            run_training=True,
            recode_map=RECODE_MAP,
            cross_validation_index=cv_index,
            fine_tune_patience=fine_tune_epochs,
            epochs=epochs,
            img_size=training_imgsz,
            batch_size=batch_size,
            cache=cache,
        )
        commit_and_push(
            dataset_label=dataset_label,
            base_dir=base_dir,
            description="Bump patch/pre-artifacts.",
        )
        detect_images_root = dst_root / "val" / "images"
        test_set_str = "val"
        test_set_part_label = f"{dataset_label}_{test_set_str}"

        run_name = run_detections(
            images_path=detect_images_root,
            dataset_version=test_set_part_label,
            model_path=Path(f"{YOLO_ROOT}/runs/train/{dataset_label}/weights/best.pt"),
            model_version=dataset_label,
            base_dir=base_dir,
            conf_thres=CONF,
            device=0,
            img_size=test_imgsz,
        )
        inferences_path = YOLO_ROOT / f"runs/detect/{run_name}/labels"
        val_inferences_roots.append(inferences_path.resolve())

    classification_metrics_for_cross_validation_set(
        dataset_prefix=fiftyone_dataset_label,
        base_dir=base_dir,
        groupings=GROUPINGS,
    )
    if init_fiftyone:
        init_fifty_one_dataset_for_cross_validation_combinations(
            dataset_label=fiftyone_dataset_label,
            classes_map=CLASSES_MAP,
            val_inferences_roots=val_inferences_roots,
            dataset_root=DATASET_ROOT,
            candidate_subset=None,
            export_to_json=True,
        )
    commit_and_push(
        dataset_label=fiftyone_dataset_label,
        base_dir=base_dir,
        description="Post-training artifacts.",
    )


def test_init_fiftyone_ds(candidate_subset: Path = None):
    delete_fiftyone_dataset(dataset_label=DATASET_LABEL)
    delete_fiftyone_dataset(dataset_label=DATASET_LABEL)
    (_, _, _, val_inferences_root) = get_labels_and_paths_tuple(
        dataset_label=DATASET_LABEL, reverse_it=False
    )
    init_fifty_one_dataset(
        dataset_label=DATASET_LABEL,
        classes_map=CLASSES_MAP,
        val_inferences_root=val_inferences_root,
        train_inferences_root=None,
        dataset_root=DATASET_ROOT,
        images_root=None,  # Use dataset_root approach
        ground_truths_root=None,  # Use dataset_root approach
        candidate_subset=candidate_subset,
    )


def run_find_errors(
    tag: str = "mistakenness",
    label_filters: Optional[List[str]] = None,
    limit: int = 64,
    dataset_label: str = None,
):
    if dataset_label is None:
        dataset_label = DATASET_LABEL
    else:
        pass
    find_errors(
        dataset_label=dataset_label,
        class_names=list(CLASSES_MAP.values()),
        tag=tag,
        limit=limit,
        processed=True,
        reverse=True,
        label_filters=label_filters,
    )
