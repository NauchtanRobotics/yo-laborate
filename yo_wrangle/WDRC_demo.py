from pathlib import Path

from yo_wrangle.wrangle import (
    copy_images_recursive_inc_yolo_annotations_by_reference_dir,
    subsample_a_directory,
)


def test_copy_by_reference():
    reference_dir = Path(
        "/home/david/RACAS/boosted/600_x_600/unmasked/Train_Charters_Towers_2021_WS"
    )
    original_images_dir = Path(
        "/home/david/RACAS/640_x_640/RACAS_CTRC_2021_sealed"
        # "/home/david/addn_repos/yolov5/runs/detect/Charters_Towers__v8e_conf8pcnt__non_agnostic2"
    )
    dst = Path("/home/david/RACAS/sealed_roads_dataset/Train_Charters_Towers_2021_WS")
    copy_images_recursive_inc_yolo_annotations_by_reference_dir(
        reference_dir=reference_dir,
        original_images_dir=original_images_dir,
        dst_sample_dir=dst,
        num=None,
        move=False,
        annotations_location="ref_yolo",  # "labels",  # "ref_yolo"
    )


def test_split_CT_WS():
    subsample_a_directory(
        src_images_root=Path(
            "/home/david/RACAS/sealed_roads_dataset/Train_Charters_Towers_2021_WS_1"
        ),
        dst_images_root=Path(
            "/home/david/RACAS/sealed_roads_dataset/Train_Charters_Towers_2021_WS_2"
        ),
        every_n_th=2,
        move=True,
    )
