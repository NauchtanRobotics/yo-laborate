from pathlib import Path
from yo_ratchet.yo_wrangle.common import get_id_to_label_map
from yo_ratchet.yo_wrangle.common import get_classes_json_path
from yo_ratchet.yo_wrangle.stats import (
    count_class_instances_in_datasets,
    count_images_having_confirmed_or_denied_boxes
)


def test_count_class_instances_in_datasets():
    base_dir = Path("/home/david/traffic_signs_dataset")  # Path(__file__).parent.parent
    sample_folders = [x for x in base_dir.iterdir() if (x.is_dir() and x.name[0] != "." and x.name[0] != "_")]
    classes_json_path = get_classes_json_path(base_dir=base_dir)
    classes_map = get_id_to_label_map(Path(f"{classes_json_path}").resolve())
    class_ids = list(classes_map.keys())
    print("\n")
    output_str = count_class_instances_in_datasets(
        data_samples=sample_folders,
        class_ids=class_ids,
        class_id_to_name_map=classes_map,
    )


def test_count_images_having_confirmed_or_denied_boxes_murrumbidgee():
    yolo_file = Path(
        r"/home/david/Downloads/Defects_murrumbidgee_council_1689755936b_edited.yolo")
    num = count_images_having_confirmed_or_denied_boxes(yolo_file=yolo_file, class_to_count=22)
    assert num > 0


def test_count_images_having_confirmed_or_denied_boxes_ai_shoving():
    yolo_file = Path(r"/home/david/Downloads/Defects_ai_shoving_1700441999_diff.yolo")
    num = count_images_having_confirmed_or_denied_boxes(yolo_file=yolo_file, class_to_count=22)
    assert num > 0


def test_count_images_having_confirmed_or_denied_boxes_lithgow():
    yolo_file = Path(
        r"/home/david/Downloads/Lithgow/Defects_lithgow_city_council_1700124175_diff.yolo")
    num = count_images_having_confirmed_or_denied_boxes(yolo_file=yolo_file, class_to_count=17)
    assert num > 0
