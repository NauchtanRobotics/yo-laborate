import psutil
from google.cloud import storage
from pathlib import Path
from typing import Optional, List, Set

YOLO_SUFFIX = ".yolo"

MIN_FREE_DISK_SPACE = 30_000

WINDOWS_LINE_ENDING = "\r\n"


def download_blobs_in_list(
    storage_client: storage.Client,
    bucket_name: str,
    image_names: Set[str],
    dst_folder: Path,
    prefix: str = ""
) -> List[Path]:
    """
    Downloads all images in set of image_names without filtering for confidence levels.

    Returns a list of fully resolved local paths to downloaded files.

    """
    local_paths: List[Path] = []
    dst_folder.mkdir(parents=True, exist_ok=True)
    for image_name in image_names:
        dst_name = dst_folder / image_name
        blob_prefix = f"{prefix}{image_name}"
        if dst_name.is_file() and dst_name.exists():
            continue
        blobs = storage_client.list_blobs(
            bucket_or_name=bucket_name, prefix=blob_prefix
        )
        blob_names = [blob.name for blob in blobs]
        try:
            blob_name = blob_names[0]
            with open(str(dst_name), "wb") as file_obj:
                gs_path = f"gs://{bucket_name}/{blob_name}"
                storage_client.download_blob_to_file(gs_path, file_obj)
            local_paths.append(dst_name)
        except:  # This file wasn't available in cloud so delete the empty local file that was created
            dst_name.unlink(missing_ok=True)
            pass
        free_disk_space = psutil.disk_usage('/').free
        if free_disk_space < MIN_FREE_DISK_SPACE:
            print("Quitting because you have less than " + str(MIN_FREE_DISK_SPACE) + " space on drive '/'.")
            break
    return local_paths


def download_training_data_for_a_subset_from_a_single_yolo_file(
    bucket_name: str,
    storage_client: storage.Client,
    yolo_file: Path,
    dst_folder: Path,
    images_prefix: str = ""
) -> List[str]:
    """
    Downloads all images in the provided yolo file WITHOUT filtering for confidence.

    """
    yolo_content = yolo_file.read_text().splitlines()
    image_names = set([row.split(" ")[0] for row in yolo_content])
    download_blobs_in_list(
        storage_client=storage_client,
        bucket_name=bucket_name,
        image_names=image_names,
        dst_folder=dst_folder,
        prefix=images_prefix,
    )
    return yolo_content


def download_training_data_for_a_subset_from_multiple_yolo_files(
    bucket_name: str,
    storage_client: storage.Client,
    yolo_files_root: Path,
    dst_folder: Path,
    images_prefix: str = ""
):
    """
    This function offers dual functionality of downloading all training images
    corresponding to object detections in one or more yolo files, plus provides an
    aggregated yolo file which can then be processed by other functions to prepare
    a dataset/subset.

    When working with `yo-laborate`, use this function to help prepare a SINGLE
    data-subset where you have reviewed annotations which are mutually exclusive
    object detections across multiple yolo files which you want to combine.

    Downloads everything in the provided yolo files WITHOUT filtering for confidence.

    :param bucket_name:
    :param storage_client:
    :param yolo_files_root:
    :param dst_folder:
    :param images_prefix:
    :return:
    """
    aggregated_content = []
    # Download images from all bounding boxes yolo text files in nominated dir <yolo_files_root>
    yolo_files = [fp for fp in yolo_files_root.iterdir() if fp.suffix == YOLO_SUFFIX]
    dst_folder = dst_folder.resolve()
    for yolo_file in yolo_files:
        if yolo_file.is_dir():
            continue
        yolo_content = download_training_data_for_a_subset_from_a_single_yolo_file(
            storage_client=storage_client,
            bucket_name=bucket_name,
            yolo_file=yolo_file,
            dst_folder=dst_folder,
            images_prefix=images_prefix,
        )
        aggregated_content.extend(yolo_content)

    # Now write all lines from individual yolo files into an aggregated yolo file
    aggregated_yolo_dst = dst_folder / f"aggregated{YOLO_SUFFIX}"
    with open(str(aggregated_yolo_dst), "w") as fp:
        for item in aggregated_content:
            fp.write("%s\n" % item)


def get_most_recent_blob_with_prefix(
    storage_client: storage.Client,
    customer_bucket_name: str,
    prefix: str,
    delimiter: Optional[str] = None
):
    """
    Assumes blobs end in a time stamp so that alphabetical sorting equates
    to sorting in order of age.

    """
    blobs = list(storage_client.list_blobs(customer_bucket_name, prefix=prefix, delimiter=delimiter))

    if len(blobs) > 0:
        return blobs[-1]
    else:
        return None


def get_latest_yolo_blobs(
    storage_client: storage.Client,
    prefix: str = "",
    skip_substring: Optional[str] = None,
    skip_buckets: Optional[List[str]] = None,
    bucket_names: Optional[List[str]] = None
):
    if bucket_names is None:
        list_buckets = storage_client.list_buckets()
        bucket_names = [bucket.name for bucket in list_buckets]
    most_recent_yolo_blobs = []
    for bucket_name in bucket_names:
        if skip_substring is not None and skip_substring in bucket_name:
            continue
        if skip_buckets is not None and bucket_name in skip_buckets:
            continue  # without doing steps below
        latest_yolo_blob = get_most_recent_blob_with_prefix(
            storage_client=storage_client,
            customer_bucket_name=bucket_name,
            prefix=prefix
        )
        if latest_yolo_blob is not None:
            most_recent_yolo_blobs.append(latest_yolo_blob)
    return most_recent_yolo_blobs


def download_all_training_data_from_buckets(
    storage_client: storage.Client,
    destination_root: Path,
    images_redirect_bucket: Optional[str] = None,
    images_path_prefix: str = "",
    skip_substring: Optional[str] = None,
    skip_buckets: Optional[List[str]] = None,
    bucket_names: Optional[List[str]] = None
):
    """
    1. Determine which buckets have images available.
    2. Download the latest yolo files from each of those buckets.
    3. Download the images associated with the object detections.

    Keeps data-subsets independent according to bucket name.

    :return:
    """
    most_recent_yolo_blobs = get_latest_yolo_blobs(
        storage_client=storage_client,
        prefix="Defects_",
        skip_substring=skip_substring,
        skip_buckets=skip_buckets,
        bucket_names=bucket_names
    )
    if images_redirect_bucket is not None:
        redirect_bucket = storage_client.bucket(images_redirect_bucket)
    else:
        redirect_bucket = None  # handled inside yolo blobs list loop below

    for blob in most_recent_yolo_blobs:
        suffix = blob.name[-4:]
        if suffix != YOLO_SUFFIX:
            continue
        print(blob.bucket.name)  # + ": " + blob.name)
        if images_redirect_bucket is None:
            images_bucket = blob.bucket
        else:
            images_bucket = redirect_bucket

        try:
            yolo_contents = blob.download_as_string()
            object_detections = [line for line in yolo_contents.splitlines()]
            image_names = set([line.decode("utf-8").split(" ")[0] for line in object_detections])
            # Create a subset folder within <destination_root>
            dst_folder = destination_root / blob.bucket.name
            dst_folder.mkdir(parents=True, exist_ok=True)
            annotations_dir = dst_folder / "YOLO_Darknet"
            annotations_dir.mkdir(exist_ok=True)
        except:
            continue
        for image_name in image_names:
            try:
                if images_redirect_bucket is not None:
                    src_blob_name = f"{blob.bucket.name}/{images_path_prefix}{image_name}"
                else:
                    src_blob_name = f"{images_path_prefix}{image_name}"

                image_blob = images_bucket.blob(src_blob_name)
                if image_blob is None:
                    continue
                # Write annotations file into YOLO_Darknet dir
                detections = [line.decode("utf-8") for line in object_detections if line.decode("utf-8").split(" ")[0] == image_name]
                detections_str = "\r\n".join(detections)
                dst_annotation = annotations_dir / f"{image_name[:-4]}.txt"

                dst_image = dst_folder / image_name

                if dst_annotation.exists() and dst_annotation.stat().st_size > 0 and dst_image.exists():
                    if dst_image.stat().st_size == 0:
                        print(f"{str(dst_image)} was empty.")
                        dst_image.unlink()
                    else:
                        continue  # save time on task resume - not need to overwrite previous downloads
                else:
                    pass  # pass through to writing and downloading below

                with open(dst_annotation, "w") as file_obj:
                    file_obj.write(detections_str)
                # Download image

                image_blob.download_to_filename(str(dst_image))
            except Exception as ex:
                # raise Exception(ex)
                print("Failed for bucket/image: ", blob.bucket.name, image_name)
                print(str(ex))
