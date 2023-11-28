from pathlib import Path
from google.cloud import storage
from yo_ratchet.yo_wrangle.gcs_interface import (
    download_all_training_data_from_buckets,
    download_training_data_for_a_subset_from_a_single_yolo_file,
    download_training_data_for_a_subset_from_multiple_yolo_files
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
