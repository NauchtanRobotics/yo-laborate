from pathlib import Path

from yo_ratchet.evaluate_binary import analyse_model_binary_metrics, analyse_model_binary_metrics_for_groups

ROOT_TEST_DATA = Path(__file__).parent / "test_data"


def test_analyse_performance():

    analyse_model_binary_metrics(
        images_root=(ROOT_TEST_DATA / "bbox_collation_7_split/val/images"),
        root_ground_truths=(ROOT_TEST_DATA / "bbox_collation_7_split/val/labels"),
        root_inferred_bounding_boxes=(
                ROOT_TEST_DATA / "Collation_7_unweighted_equal_fitness_scale40pcnt/labels"
        ),
        class_name_list_path=Path(
            "C:\\Users\\61419\\OpenLabeling\\main\\class_list.txt"
        ),
        print_first_n=15,
        dst_csv=Path(__file__).parent / "results.csv"
    )


def test_analyse_model_binary_metrics_for_groups():
    analyse_model_binary_metrics_for_groups(
        images_root=(ROOT_TEST_DATA / "bbox_collation_7_split/val/images"),
        root_ground_truths=(ROOT_TEST_DATA / "bbox_collation_7_split/val/labels"),
        root_inferred_bounding_boxes=(
                ROOT_TEST_DATA / "Collation_7_unweighted_equal_fitness_scale40pcnt/labels"
        ),
        class_names_path=Path(
            "C:\\Users\\61419\\OpenLabeling\\main\\class_list.txt"
        ),
        groupings={
            "Risk Defects": [3, 4],
            "Cracking": [0, 1, 2, 16],
        },
    )
