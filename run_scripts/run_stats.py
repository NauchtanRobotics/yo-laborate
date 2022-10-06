from pathlib import Path

from yo_ratchet.yo_wrangle.stats import count_class_instances_in_datasets

# Example Usage:
#
# def test_count_class_instances_in_datasets():
#     sample_folders = [
#         (
#             Path(
#                 "/home/david/RACAS/boosted/600_x_600/unmasked/Caboone_10pcnt_AP_LO_LG_WS"
#             ),
#             None,
#         ),
#     ]
#     class_ids = list(classes_map.keys())
#
#     output_str = count_class_instances_in_datasets(
#         data_samples=sample_folders,
#         class_ids=class_ids,
#         class_id_to_name_map=classes_map,
#     )
#     print(output_str)
