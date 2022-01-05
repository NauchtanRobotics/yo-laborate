"""
Standard dataset building steps::

    1. Merge images from <src_images> folder with detected "labels" according to reference
       images in "cropped". Source images are flattened into <dst_images> folder and
       annotations end up in <dst_images>/YOLO_darknet folder as required for editing
       bounding boxes when using OpenLabeling library. Use::

        -> copy_recursive_images_and_yolo_annotations_by_cropped_image_reference()

    2. Uniformly sample positive YOLO detections from a project. Use::

        -> subsample_a_directory()

    3. Extract all unique samples for specific classes from confident detections
       to complement previous samples taken from YOLO detections (perhaps at lower confidence).
       Use::

        -> prepare_unique_dataset_from_detections()

    4. Collate multiple samples from one project into a common directory. Use::

        -> collate_additional_sample()

    5. Collate samples taken from various projects and split into train and validation
       data sets. Use::

        -> collate_image_and_annotation_subsets()

"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from yo_ratchet.dataset_versioning.tag import get_path_for_best_pretrained_model
from yo_ratchet.dataset_versioning import commit_and_push
from yo_ratchet.yo_wrangle.stats import count_class_instances_in_datasets
from yo_ratchet.yo_wrangle.common import (
    get_all_jpg_recursive,
    get_all_txt_recursive,
    YOLO_ANNOTATIONS_FOLDER_NAME,
    get_config_items,
)


def copy_images_recursive_inc_yolo_annotations_by_reference_dir(
    reference_dir: Path,  # Can be the same as the original_images_dir when moving imgs
    original_images_dir: Path,
    dst_sample_dir: Path,
    num: Optional[int] = None,
    move: bool = False,
    annotations_location: str = "yolo",  # "yolo" | "ref_yolo" | "labels"
):
    """
    Used mainly to copy images into a sample folder based on cropped YOLO
    detections.

    Copies a sample of original images and their corresponding annotation files.
    Moves images to <dst_sample_dir>/ and annotations to
    <dst_sample_dir>/YOLO_darknet/

    The images folder structure is the destination is FLATTENED.

    Reference dir can be different to the original_images_dir when a reference dir
    has been cleaned up (e.g. images with unsealed roads or certain defect types
    have been deleted) but the reference dir does not contain the annotations so
    this directory is not directly helpful to move.

    Optional params that are particularly useful for splitting into train, val
    datasets::
        * :param num: int, Breaks after num images
        * :param move: bool, Move instead of copy.

    n.b. This function assumes by default that there are annotations in
    <original_images_dir>/YOLO_darknet (compatible with OpenLabel). However,
    using the annotations_location param this assumption can be changed to
    be compatible with the folder structure produce by YOLOv5 detections,
    i.e.:
        <reference_dir>/"cropped"
        <reference_dir>/"labels"

    """
    if (
        annotations_location == "yolo"
    ):  # YOLO_darknet dir nested in the original_images dir
        src_annotations_dir = original_images_dir / YOLO_ANNOTATIONS_FOLDER_NAME
    elif (
        annotations_location == "ref_yolo"
    ):  # YOLO_darknet dir nested in withing the reference_dir directory
        src_annotations_dir = reference_dir / YOLO_ANNOTATIONS_FOLDER_NAME
    elif (
        annotations_location == "labels"
    ):  # E.g. In a "labels" directory which is beside
        src_annotations_dir = reference_dir / annotations_location
    else:
        raise Exception(
            f"Annotations dir structure not allowed: {str(reference_dir / annotations_location)}"
        )

    dst_annotations_dir = dst_sample_dir / YOLO_ANNOTATIONS_FOLDER_NAME
    dst_annotations_dir.mkdir(parents=True, exist_ok=True)
    for i, reference_image_path in enumerate(
        sorted(get_all_jpg_recursive(img_root=reference_dir))
    ):
        if num and i >= num:
            break
        original_image_path = original_images_dir / reference_image_path.name
        if (
            annotations_location == "labels" and not original_image_path.exists()
        ):  # Image is associated with > 1 crops.
            original_image_path_stem = original_image_path.stem[:-1]
            original_image_path = (
                original_image_path.parent / f"{original_image_path_stem}.jpg"
            )
            if (
                not original_image_path.exists()
            ):  # There were 10 or more crops, so trim stem by one more char.
                original_image_path_stem = original_image_path.stem[:-1]
                original_image_path = (
                    original_image_path.parent / f"{original_image_path_stem}.jpg"
                )
                if not original_image_path.exists():
                    print(f"Does not exist: {str(original_image_path)}")
                    continue

        dst_image_path = dst_sample_dir / original_image_path.name
        if move:
            shutil.move(src=original_image_path, dst=dst_image_path)
        else:
            shutil.copy(src=original_image_path, dst=dst_image_path)

        src_annotations_path = src_annotations_dir / f"{original_image_path.stem}.txt"
        if not src_annotations_path.exists():
            continue
        dst_annotations_path = (
            # dst_sample_dir
            # / YOLO_ANNOTATIONS_FOLDER_NAME
            dst_annotations_dir
            / f"{original_image_path.stem}.txt"
        )
        if move:
            shutil.move(src=src_annotations_path, dst=dst_annotations_path)
        else:
            shutil.copy(src=src_annotations_path, dst=dst_annotations_path)

    if annotations_location == "labels":
        for annotations_path in get_all_txt_recursive(root_dir=dst_annotations_dir):
            new_lines = []
            with open(str(annotations_path), mode="r+") as f:
                lines = f.readlines()
                for line in lines:
                    line_split = line.split(" ")
                    num_boxes = (len(line_split) - 1) // 4
                    if num_boxes > 1:
                        raise Exception("Take a looky here")
                    if len(line_split) > 5:
                        new_line = " ".join(line_split[0:-1])
                        new_lines.append(f"{new_line}\n")

            if len(line_split) > 5:
                with open(str(annotations_path), mode="w") as f:
                    f.writelines(new_lines)


def subsample_a_directory(
    src_images_root: Path,
    dst_images_root: Path,
    every_n_th: int = 6,
    move: bool = False,
):
    """
    Only intended to work on directory after annotations have been moved to "YOLO_darknet"
    folder (won't work directly on detection results where annotations are in "labels".
    Best to run copy_images_recursive_inc_yolo_annotations_by_reference_dir() on yolo
    detect results first.

    Similar to importer.subsample_a_directory() but additionally copies annotations.

    """
    dst_images_root.mkdir(parents=True)
    dst_annotations_root = dst_images_root / YOLO_ANNOTATIONS_FOLDER_NAME
    dst_annotations_root.mkdir(parents=True)
    for i, image_path in enumerate(get_all_jpg_recursive(img_root=src_images_root)):
        if i % every_n_th != 0:
            continue
        outfile = dst_images_root / image_path.name
        if move:  # Move the image
            shutil.move(src=str(image_path), dst=str(outfile))
        else:
            shutil.copyfile(src=str(image_path), dst=str(outfile))

        src_annotations_file = (
            src_images_root / YOLO_ANNOTATIONS_FOLDER_NAME / f"{image_path.stem}.txt"
        )
        dst_annotations_file = dst_annotations_root / f"{image_path.stem}.txt"
        if move:  # Move the annotation
            shutil.move(src=str(src_annotations_file), dst=str(dst_annotations_file))
        else:
            shutil.copyfile(
                src=str(src_annotations_file), dst=str(dst_annotations_file)
            )


def copy_detect_folder_recursively_as_reference_then_subsample(
    reference_dir: Path,
    original_images_dir: Path,
    dst_sample_dir: Path,
    annotations_location="labels",
    every_n_th=6,
):
    temp_file = tempfile.mkdtemp()
    temp_path = Path(temp_file)
    copy_images_recursive_inc_yolo_annotations_by_reference_dir(
        reference_dir=reference_dir,
        original_images_dir=original_images_dir,
        dst_sample_dir=temp_path,
        num=None,
        move=False,
        annotations_location=annotations_location,
    )
    subsample_a_directory(
        src_images_root=temp_path,
        dst_images_root=dst_sample_dir,
        every_n_th=every_n_th,
    )
    shutil.rmtree(str(temp_path))


def collate_image_and_annotation_subsets(
    samples_required: List[Tuple[Path, Optional[int]]],
    dst_folder: Path,
    keep_class_ids: Optional[List[int]] = None,
    skip_class_ids: Optional[List[int]] = None,
):
    """
    Copies a sample of original images and their corresponding annotation files
    to <dst_folder>/ and <dst_folder>/YOLO_darknet/ respectively.

    This function assumes that there are annotations in original_images_dir/YOLO_darknet
    and will replicate this structure in the destination folder.

    :param samples_required: List of tuples of Path to folder to be sampled, qty images to sample
    :param dst_folder:       Destination for collated sample images/annotations.
    :param keep_class_ids:   List of class ids to for which annotations should be removed.To keep all
                             class ids, simply set to None (default).
    :param skip_class_ids:   List of class ids to for which annotations should be kept.

    """
    dst_folder.mkdir(exist_ok=True)
    for original_images_dir, sample_size in samples_required:
        assert (
            original_images_dir.exists()
        ), f"Subset path not found: {str(original_images_dir)}"
        src_annotations_dir = original_images_dir / YOLO_ANNOTATIONS_FOLDER_NAME
        dst_annotations_folder = dst_folder / YOLO_ANNOTATIONS_FOLDER_NAME
        dst_annotations_folder.mkdir(exist_ok=True)
        original_image_paths = sorted(
            get_all_jpg_recursive(img_root=original_images_dir)
        )
        assert (
            len(original_image_paths) > 0
        ), f"Subset path has no jpg files: {str(original_images_dir)}"
        for i, original_image_path in enumerate(original_image_paths):
            if sample_size and i >= sample_size:
                break
            dst_image_path = dst_folder / original_image_path.name
            # if dst_image_path.exists():
            #     dst_image_path = dst_folder / f"{original_image_path.stem}zzz{original_image_path.suffix}"
            if dst_image_path.exists():
                print(f"File name is not unique, skipping {str(dst_image_path.name)}")
                continue
            shutil.copy(src=original_image_path, dst=dst_image_path)

            src_annotations_path = (
                src_annotations_dir / f"{original_image_path.stem}.txt"
            )
            dst_annotations_path = (
                dst_folder
                / YOLO_ANNOTATIONS_FOLDER_NAME
                / f"{original_image_path.stem}.txt"
            )
            if src_annotations_path.exists():
                shutil.copy(src=src_annotations_path, dst=dst_annotations_path)
            else:
                print(f"Annotation file not found: {str(src_annotations_path)}")

        if keep_class_ids or skip_class_ids:
            filter_dataset_for_classes(
                annotations_dir=dst_annotations_folder,
                keep_class_ids=keep_class_ids,
                skip_class_ids=skip_class_ids,
            )


def split_yolo_train_dataset_every_nth(
    src_images_root: Path,
    dst_dataset_root: Path,
    every_n_th: int = 5,
    cross_validation_index: int = 0,
):
    """
    Copies images and annotations from <src_images_root> and <src_images_root>/YOLO_darknet
    respectively to a structure expected by yolov5/, as follows::

        <dst_dataset_root>/images/train/
        <dst_dataset_root>/images/val/
        <dst_dataset_root>/labels/train/
        <dst_dataset_root>/labels/val/

    """
    dst_dataset_root.mkdir(exist_ok=True, parents=True)
    dst_train_root = dst_dataset_root / "train"
    dst_train_root.mkdir()
    dst_val_root = dst_dataset_root / "val"
    dst_val_root.mkdir()

    dst_train_images_root = dst_train_root / "images"
    dst_train_images_root.mkdir()
    dst_train_labels_root = dst_train_root / "labels"
    dst_train_labels_root.mkdir()

    dst_val_images_root = dst_val_root / "images"
    dst_val_images_root.mkdir()
    dst_val_labels_root = dst_val_root / "labels"
    dst_val_labels_root.mkdir()

    for i, image_path in enumerate(get_all_jpg_recursive(img_root=src_images_root)):
        if i < cross_validation_index:
            continue
        src_annotations_file = (
            src_images_root / YOLO_ANNOTATIONS_FOLDER_NAME / f"{image_path.stem}.txt"
        )

        if (i - cross_validation_index) % every_n_th != 0:
            dst = dst_train_root
        else:
            dst = dst_val_root

        dst_image_path = dst / "images" / image_path.name
        shutil.copy(src=str(image_path), dst=str(dst_image_path))

        dst_annotations_file = dst / "labels" / src_annotations_file.name
        if src_annotations_file.exists():
            shutil.copy(src=str(src_annotations_file), dst=str(dst_annotations_file))
        else:
            print(f"Annotation file not found: {str(src_annotations_file)}")


def check_train_val_are_unique(dataset_path: Path):
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"
    train_image_names = [x.name for x in get_all_jpg_recursive(img_root=train_path)]
    val_image_names = [x.name for x in get_all_jpg_recursive(img_root=val_path)]
    assert len(set(val_image_names) & set(train_image_names)) == 0


def collate_and_split(
    subsets_included: List[Tuple[Path, Optional[int]]],
    dst_root: Path,
    every_n_th: int = 1,  # for the validation subset.
    keep_class_ids: Optional[List] = None,  # None actually means keep all classes
    skip_class_ids: Optional[List] = None,
    cross_validation_index: int = 0,
):
    if dst_root.exists():
        shutil.rmtree(str(dst_root))
    temp_dir = tempfile.mkdtemp()
    collate_image_and_annotation_subsets(
        samples_required=subsets_included,
        dst_folder=Path(temp_dir),
        skip_class_ids=skip_class_ids,
        keep_class_ids=keep_class_ids,
    )
    split_yolo_train_dataset_every_nth(
        src_images_root=Path(temp_dir),
        dst_dataset_root=dst_root,
        every_n_th=every_n_th,
        cross_validation_index=cross_validation_index,
    )
    check_train_val_are_unique(dataset_path=dst_root)
    shutil.rmtree(temp_dir)


def _get_hits_for_annotations_in_classes(
    classes: List[int],
    src_images_dir: Path,
) -> List[str]:
    """
    Returns list of image filename stems for annotation files which contain
    at least one of the classes specified in the param `classes`.

    """
    src_annotations_dir = src_images_dir / YOLO_ANNOTATIONS_FOLDER_NAME
    hit_list: List[str] = []
    for src_annotation_path in get_all_txt_recursive(root_dir=src_annotations_dir):
        with open(str(src_annotation_path)) as f:
            lines = f.readlines()
        for line in lines:
            class_idx = int(line.strip().split(" ")[0])
            if class_idx in classes:
                hit_list.append(src_annotation_path.stem)
                break
    return hit_list


def delete_redundant_samples(
    sample_folder_to_clean: Path,
    other_sample_folders: List[Path],
):
    reference_image_paths = []
    for sample_folder in other_sample_folders:
        for image_path in get_all_jpg_recursive(sample_folder):
            reference_image_paths.append(image_path.name)
    unique_images = []
    for image_path in get_all_jpg_recursive(sample_folder_to_clean):
        image_name = image_path.name
        if image_name in reference_image_paths:
            image_path.unlink()
        if image_name in unique_images:
            image_path.unlink()
        else:
            unique_images.append(image_name)


def prepare_unique_dataset_from_detections(
    reference_dir: Path,
    original_images_dir: Path,
    dst_sample_dir: Path,
    class_names: List[str],
    other_sample_folders: List[Path],
):
    """
    1. Copies a detection directory to temporary location filtering for selected class folders.
    2. Run copy_by_reference to another destination directory
    3. Run delete_redundant_samples() applied to the destination directory.

    """
    temp_file = tempfile.mkdtemp()
    temp_path = Path(temp_file)
    src_folder = reference_dir / "labels"
    dst_folder = Path(temp_path) / "labels"
    shutil.copytree(src=str(src_folder), dst=str(dst_folder))
    src_crops_dir = reference_dir / "crops"
    dst_crops_dir = temp_path / "crops"
    dst_crops_dir.mkdir()
    for folder in src_crops_dir.iterdir():
        if folder.is_dir() and folder.name in class_names:
            dst_folder = temp_path / "crops" / folder.name
            shutil.copytree(src=str(folder), dst=str(dst_folder))

    copy_images_recursive_inc_yolo_annotations_by_reference_dir(
        reference_dir=Path(temp_path),
        original_images_dir=original_images_dir,
        dst_sample_dir=Path(dst_sample_dir),
        num=None,
        move=False,
        annotations_location="labels",
    )
    shutil.rmtree(temp_path)

    delete_redundant_samples(
        sample_folder_to_clean=dst_sample_dir,
        other_sample_folders=other_sample_folders,
    )


def filter_dataset_for_classes(
    annotations_dir: Path,
    keep_class_ids: Optional[List[int]] = None,
    skip_class_ids: Optional[List[int]] = None,
):
    """
    WARNING::
        Designed to be used on copies - permanently deletes annotations.

    """

    for annotations_path in get_all_txt_recursive(root_dir=annotations_dir):
        with open(str(annotations_path), "r") as file:
            annotation_lines = file.readlines()

        new_lines = []
        for annotation in annotation_lines:
            class_id = int(annotation.strip().split(" ")[0])
            if skip_class_ids and class_id in skip_class_ids:
                continue
            elif keep_class_ids and class_id not in keep_class_ids:
                continue
            new_lines.append(annotation)

        with open(str(annotations_path), "w") as file:
            file.writelines(new_lines)


def collate_additional_sample(
    existing_sample_dir: Path,
    additional_sample_dir: Path,
    dst_folder: Path,
):
    """
    Cannot be done within prepare_unique_dataset_from_detections() as the bounding
    boxes need to be manually established.

    """
    sample_folders = [
        (existing_sample_dir, None),
        (additional_sample_dir, None),
    ]
    collate_image_and_annotation_subsets(
        samples_required=sample_folders,
        dst_folder=dst_folder,
        keep_class_ids=None,
    )


def prepare_dataset_and_train(
    classes_map: Dict[int, str],
    subsets_included: List,
    dst_root: Path,
    every_n_th: int,
    keep_class_ids: Optional[List[int]],
    skip_class_ids: Optional[List[int]],
    base_dir: Path,
    run_training: bool = True,
    cross_validation_index: int = 0,
):
    class_ids = list(classes_map.keys())
    output_str = count_class_instances_in_datasets(
        data_samples=subsets_included,
        class_ids=class_ids,
        class_id_to_name_map=classes_map,
    )
    model_instance = dst_root.name

    with open(f"{model_instance}_classes_support.txt", "w") as f_out:
        f_out.write(output_str)

    collate_and_split(
        subsets_included=subsets_included,
        dst_root=dst_root,
        every_n_th=every_n_th,
        keep_class_ids=keep_class_ids,
        skip_class_ids=skip_class_ids,
        cross_validation_index=cross_validation_index,
    )
    """Add actual classes support after filtering"""
    final_subsets_included = [
        ((dst_root / "train"), None),
        ((dst_root / "val"), None),
    ]
    output_str = count_class_instances_in_datasets(
        data_samples=final_subsets_included,
        class_ids=class_ids,
        class_id_to_name_map=classes_map,
    )
    output_str = "\n" + output_str
    with open(f"{model_instance}_classes_support.txt", "a") as f_out:
        f_out.write(output_str)

    class_names = [classes_map[class_id] for class_id in class_ids]
    yaml_text = f"""train: {str(dst_root)}/train/images/
val: {str(dst_root)}/val/images/
nc: {len(class_ids)}
names: {class_names}"""

    """ Write dataset.yaml locally. """
    with open(f"{model_instance}_dataset_yaml.yaml", "w") as f_out:
        f_out.write(yaml_text)

    commit_and_push(
        dataset_label=dst_root.name,
        base_dir=base_dir,
    )

    """ Write dataset.yaml in DST folder."""
    dst_dataset_path = dst_root / "dataset.yaml"
    with open(f"{str(dst_dataset_path)}", "w") as f_out:
        f_out.write(yaml_text)

    (
        python_path,
        yolo_base_dir,
        cfg_path,
        _,
        hyp_path,
        _,
        _,
    ) = get_config_items(base_dir)
    weights_path, fine_tune = get_path_for_best_pretrained_model(base_dir=base_dir)
    if fine_tune:
        patience = 5
    else:
        patience = 50
    train_script = str(Path(yolo_base_dir) / "train.py")
    pytorch_cmd = [
        python_path,
        train_script,
        "--img=640",
        "--batch=62",
        "--workers=4",
        "--device=0,1",
        f"--cfg={cfg_path}",
        "--epochs=300",
        f"--data={str(dst_dataset_path)}",
        f"--weights={weights_path}",
        f"--hyp={hyp_path}",
        f"--name={model_instance}",
        f"--patience={str(patience)}",
        "--cache",
        "--freeze=3",
    ]
    if fine_tune:
        pytorch_cmd.append("--start-epoch=295")
    else:
        pass

    train_cmd_str = " ".join(pytorch_cmd)
    with open(f"{model_instance}_train_cmd.txt", "w") as f_out:
        f_out.write(train_cmd_str)

    if run_training:
        subprocess.check_call(
            pytorch_cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=str(Path(yolo_base_dir).parent),
        )


def reverse_train(
    classes_map: Dict[int, str],
    base_dir: Path,
    dst_root: Path,
):
    commit_and_push(
        dataset_label=dst_root.name,
        base_dir=base_dir,
    )
    class_ids = list(classes_map.keys())
    class_names = [classes_map[class_id] for class_id in class_ids]
    yaml_text = f"""train: {str(dst_root)}/val/images/
val: {str(dst_root)}/train/images/
nc: {len(class_ids)}
names: {class_names}"""

    """ Write dataset.yaml in DST folder."""
    dst_dataset_path = dst_root / "reverse_dataset.yaml"
    with open(f"{str(dst_dataset_path)}", "w") as f_out:
        f_out.write(yaml_text)

    python_path, yolo_root, cfg_path, weights_path, hyp_path, _, _ = get_config_items(
        base_dir
    )
    model_instance = f"{dst_root.name}_reverse"
    train_script = Path(yolo_root) / "train.py"
    pytorch_cmd = [
        python_path,
        f"{str(train_script)}",
        "--img=640",
        "--batch=50",
        "--workers=4",
        "--device=0,1",
        f"--cfg={cfg_path}",
        "--epochs=300",
        f"--data={str(dst_dataset_path)}",
        f"--weights={weights_path}",
        f"--hyp={hyp_path}",
        f"--name={model_instance}",
        "--patience=50",
        "--cache",
    ]
    print("\n")
    print(" ".join(pytorch_cmd))
    subprocess.check_call(
        pytorch_cmd,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        cwd=yolo_root,
    )


def run_detections(
    images_path: Path,
    dataset_version: str,
    model_path: Path,
    model_version: str,
    base_dir: Path,
    conf_thres: float = 0.1,
    device: int = 0,
):
    results_name = f"{dataset_version}__{model_version}_conf{int(conf_thres * 100)}pcnt"
    python_path, yolo_root, _, _, _, _, _ = get_config_items(base_dir)
    detect_script = Path(yolo_root) / "detect.py"
    pytorch_cmd = [
        python_path,
        f"{str(detect_script)}",
        f"--source={str(images_path)}",
        f"--weights={model_path}",
        "--img=640",
        f"--device={device}",
        f"--name={results_name}",
        "--save-txt",
        "--save-conf",
        "--nosave",
        "--agnostic-nms",
        f"--iou-thres=0.55",
        f"--conf-thres={conf_thres}",
    ]
    print(" ".join(pytorch_cmd))
    subprocess.check_call(
        pytorch_cmd,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        cwd=yolo_root,
    )
