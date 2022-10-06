from pathlib import Path

from yo_ratchet.yo_valuate.as_classification import analyse_model_binary_metrics


def test_analyse_performance():
    test_data_root = Path(__file__).parent / "test_data"
    analyse_model_binary_metrics(
        images_root=Path(
            "/home/david/addn_repos/yolov5/datasets/bbox_collation_7_split/val/images"
        ),
        root_ground_truths=Path(
            "/home/david/addn_repos/yolov5/datasets/bbox_collation_7_split/val/labels"
        ),
        root_inferred_bounding_boxes=Path(
            "/home/david/addn_repos/yolov5/runs/detect/Collation_7_unweighted_equal_fitness_scale40pcnt2/labels"
        ),
        class_name_list_path=Path(
            "C:\\Users\\61419\\OpenLabeling\\main\\class_list.txt"
        ),
        num_classes=20,
        dst_csv=Path(__file__).parent / "results.csv"
    )
