from pathlib import Path
from google.cloud import storage
from yo_ratchet.yo_wrangle.gcs_interface import (
    download_all_training_data_from_buckets,
    download_training_data_for_a_subset_from_a_single_yolo_file,
    download_training_data_for_a_subset_from_multiple_yolo_files,
    download_all_blobs_in_bucket,
    download_file_from_gcs_using_public_url,
    download_blobs_in_list
)

JSON_CREDENTIALS_PATH = Path(__file__).parent.parent / "GOOGLE_APPLICATION_CREDENTIALS.json"

cred_path = str(JSON_CREDENTIALS_PATH)
# storage_client = storage.Client.create_anonymous_client()
storage_client = storage.Client.from_service_account_json(
        project="sacred-bonus-274204",
        json_credentials_path=cred_path
    )


def test_download_all_training_data_from_buckets():
    download_all_training_data_from_buckets(
        storage_client=storage_client,
        destination_root=Path("/media/david/Samsung_T8/bulk_download_2023_07"),
        images_redirect_bucket="racas_ai",
        images_path_prefix="unprocessed_images/",
        bucket_names=["western_downs_regional_council"]
    )


def test_download_training_data_for_a_subset_from_a_single_yolo_file():
    download_training_data_for_a_subset_from_a_single_yolo_file(
        bucket_name="racas_ai",
        storage_client=storage_client,
        yolo_file=Path("/home/david/Downloads/Defects_ai_shoving_1700441999_diff.yolo"),
        dst_folder=Path("/home/david/production/ai_shoving_0"),
        images_prefix="ai-shoving/unprocessed_images/"
    )


def test_download_training_data_for_a_subset_from_multiple_yolo_files():
    download_training_data_for_a_subset_from_multiple_yolo_files(
        bucket_name="racas_ai",
        storage_client=storage_client,
        yolo_files_root=Path("/home/david/Downloads/Lithgow"),
        dst_folder=Path("/home/david/production/Lithgow_2023_1"),
        images_prefix="lithgow_city_council/unprocessed_images/"
    )


def test_download_all_images_in_tablelands_bucket():
    download_all_blobs_in_bucket(
        storage_client=storage_client,
        bucket_name="racas_ai",
        prefix="tablelands_regional_council/unprocessed_images/",
        dst_root=Path("/media/david/Samsung_T8/bulk_download_2023_07/tablelands_regional_council")
    )


def test_download_all_images_in_murrumbidgee_bucket():
    download_all_blobs_in_bucket(
        storage_client=storage_client,
        bucket_name="racas_ai",
        prefix="murrumbidgee_council/unprocessed_images/",
        dst_root=Path("/media/david/Samsung_T8/bulk_download_2023_07/murrumbidgee_council")
    )


def test_download_file_from_gcs_using_public_url():
    download_file_from_gcs_using_public_url(
        bucket_name="murrumbidgee_council",
        file_name="Photo_2023_Apr_20_16_37_30_079_j.jpg",
        dst_root=Path("/home/david/production/test_council")
    )


def test_download_blobs_in_list():
    image_names = {
        "Photo_2023_Jun_15_14_04_18_943_j.jpg",
        "Photo_2023_Jul_02_09_05_05_268_w.jpg",
        "Photo_2023_Jun_13_14_40_12_146_j.jpg",
        "Photo_2023_Jun_13_14_40_12_582_j.jpg",
        "Photo_2023_Jun_13_14_40_13_051_j.jpg",
        "Photo_2023_Jun_13_14_40_13_462_j.jpg",
        "Photo_2023_Jun_13_14_40_14_011_j.jpg",
        "Photo_2023_Jun_13_14_40_14_482_j.jpg",
        "Photo_2023_Jun_13_14_40_15_119_j.jpg",
        "Photo_2023_Jun_13_14_40_15_567_j.jpg",
        "Photo_2023_Jun_13_14_40_16_041_j.jpg",
        "Photo_2023_Jun_21_14_47_03_241_j.jpg",
        "Photo_2023_Jun_21_14_45_53_598_j.jpg",
        "Photo_2023_Jun_21_14_45_56_811_j.jpg",
        "Photo_2023_Jun_21_14_45_57_277_j.jpg",
        "Photo_2023_Jun_21_14_40_35_274_j.jpg",
        "Photo_2023_Jun_13_14_40_47_762_j.jpg",
        "Photo_2023_Jun_13_15_06_15_889_j.jpg",
        "Photo_2023_Jun_13_15_06_16_983_j.jpg",
        "Photo_2023_Jun_13_14_55_41_467_j.jpg",
        "Photo_2023_Jun_07_10_57_59_375_j.jpg",
        "Photo_2023_Jun_15_14_30_43_348_j.jpg",
        "Photo_2023_Jun_15_13_03_45_161_j.jpg",
        "Photo_2023_Jun_15_14_04_18_943_j.jpg",
        "Photo_2023_Jul_02_09_05_05_268_w.jpg",
        "Photo_2023_Jun_13_14_40_12_146_j.jpg",
        "Photo_2023_Jun_13_14_40_12_582_j.jpg",
        "Photo_2023_Jun_13_14_40_13_051_j.jpg",
        "Photo_2023_Jun_13_14_40_13_462_j.jpg",
        "Photo_2023_Jun_13_14_40_14_011_j.jpg",
        "Photo_2023_Jun_13_14_40_14_482_j.jpg",
        "Photo_2023_Jun_13_14_40_15_119_j.jpg",
        "Photo_2023_Jun_13_14_40_15_567_j.jpg",
        "Photo_2023_Jun_13_14_40_16_041_j.jpg",
        "Photo_2023_Jun_21_14_47_03_241_j.jpg",
        "Photo_2023_Jun_21_14_45_53_598_j.jpg",
        "Photo_2023_Jun_21_14_45_56_811_j.jpg",
        "Photo_2023_Jun_21_14_45_57_277_j.jpg",
        "Photo_2023_Jun_21_14_40_35_274_j.jpg",
        "Photo_2023_Jun_13_14_40_47_762_j.jpg",
        "Photo_2023_Jun_13_15_06_15_889_j.jpg",
        "Photo_2023_Jun_13_15_06_16_983_j.jpg",
        "Photo_2023_Jun_13_14_55_41_467_j.jpg",
        "Photo_2023_Jun_07_10_57_59_375_j.jpg",
        "Photo_2023_Jun_15_14_30_43_348_j.jpg",
        "Photo_2023_Jun_15_13_03_45_161_j.jpg"
    }
    download_blobs_in_list(
        storage_client=storage_client,
        bucket_name="racas_ai",
        image_names=image_names,
        dst_folder=Path("/home/david/production/Coonamble_missing"),
        prefix="coonamble_shire_council/unprocessed_images/"
    )
