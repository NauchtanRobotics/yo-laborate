from pathlib import Path

from yo_ratchet.yo_wrangle.mine import prepare_training_data_for_confirmed_or_denied_boxes_in_yolo_file, \
    prepare_training_data_subset_from_reviewed_yolo_file


def test_prepare_training_data_for_confirmed_or_denied_boxes_in_yolo_file():
    yolo_file = Path(
        "/home/david/production/Lithgow_2023_1/aggregated.yolo")
    assert yolo_file.exists()
    images_archive_root = Path("/home/david/production/Lithgow_2023_1")
    assert images_archive_root.exists()
    len_images = len(list(images_archive_root.rglob("*.jpg")))
    assert len_images > 0
    prepare_training_data_for_confirmed_or_denied_boxes_in_yolo_file(
        images_archive_dir=images_archive_root,
        yolo_file=yolo_file,
        dst_images_dir=Path("/home/david/production/sealed_roads_dataset/Lithgow_2023_1"),
        copy_all_src_images=False,
        move=False,  # Do dry run before changing this parameter to True
    )


def test_prepare_training_data_for_confirmed_or_denied_boxes_in_yolo_file_ai_shoving():
    yolo_file = Path("/home/david/Downloads/Defects_ai_shoving_1700441999_diff.yolo")
    assert yolo_file.exists()
    images_archive_root = Path("/home/david/production/ai_shoving_0")
    assert images_archive_root.exists()
    len_images = len(list(images_archive_root.rglob("*.jpg")))
    assert len_images > 0
    prepare_training_data_for_confirmed_or_denied_boxes_in_yolo_file(
        images_archive_dir=images_archive_root,
        yolo_file=yolo_file,
        dst_images_dir=Path("/home/david/production/sealed_roads_dataset/ai_shoving_0"),
        copy_all_src_images=False,
        move=True,  # Do dry run before changing this parameter to True
    )


def test_prepare_training_data_subset_from_reviewed_yolo_file():
    """
    This tests an older function which had a probability_thresh_coefficient to weed out any
    low confidence bounding boxes.

    :return:
    """
    yolo_file = Path(
        "/media/david/Carol_sexy/Defects_murrumbidgee_council_1689755936b_edited.yolo")
    assert yolo_file.exists()
    images_archive_root = Path("/media/david/Samsung_T8/bulk_download_2023_07/murrumbidgee_council")
    assert images_archive_root.exists()
    len_images = len(list(images_archive_root.rglob("*.jpg")))
    assert len_images > 0
    prepare_training_data_subset_from_reviewed_yolo_file(
        images_archive_dir=images_archive_root,
        yolo_file=yolo_file,
        dst_images_dir=Path("/home/david/production/sealed_roads_dataset/Murrumbidgee_2023"),
        classes_json_path=Path("/home/david/production/sealed_roads_dataset/classes.json"),
        copy_all_src_images=False,
        move=False,  # Do dry run before changing this parameter to True
        probability_thresh_coefficient=0.0001  # include all bounding boxes that were not denied
    )
