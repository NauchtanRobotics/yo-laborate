import numpy as np
import pandas
from cv2 import cv2
from pathlib import Path
from typing import List, Optional, Tuple
from cv2.cv2 import VideoWriter_fourcc

from yo_ratchet.yo_wrangle.common import (
    get_all_jpg_recursive,
    ORANGE,
    GREEN,
    RED,
    PURPLE,
)

COLOUR_MAPPING = {
    GREEN: (0, 255, 0),
    RED: (0, 0, 255),
    ORANGE: (0, 165, 255),
    PURPLE: (255, 0, 255),
}
TEXT_POSITION_MAPPING = {
    "top_left": {
        GREEN: [0.05, 0.05],
        ORANGE: [0.05, 0.1],
        RED: [0.05, 0.15],
        PURPLE: [0.05, 0.2],
    },
    "top_centre": {
        GREEN: [0.45, 0.05],
        ORANGE: [0.45, 0.1],
        RED: [0.45, 0.15],
        PURPLE: [0.05, 0.2],
    },
}
LINE_THICKNESS = 2
MARGINAL_PROB_THRESH = 0.24

# CLIENT SPECIFIC DATA
LEGEND_POSITION = "top_left"
LABEL_MAPPING = {
    "3": {
        "colour": RED,
        "label": "",
    },
    "4": {
        "colour": RED,
        "label": "",
    },
    "0": {
        "colour": ORANGE,
        "label": "",
    },
    "1": {
        "colour": ORANGE,
        "label": "",
    },
    "2": {
        "colour": ORANGE,
        "label": "",
    },
    "5": {
        "colour": GREEN,
        "label": "",
    },
    "6": {
        "colour": GREEN,
        "label": "",
    },
    "10": {
        "colour": GREEN,
        "label": "",
    },
    "11": {
        "colour": ORANGE,
        "label": "",
    },
    "12": {
        "colour": ORANGE,
        "label": "",
    },
    "16": {
        "colour": ORANGE,
        "label": "",
    },
    "17": {
        "colour": ORANGE,
        "label": "",
    },
}


def draw_polygon_on_image(
    image: np.ndarray,
    yolo_box: List[List[float]],
    dst_path: Path = None,
    label: Optional[str] = None,
    colour: str = "green",
    banner_height: Optional[int] = None,
):
    """
    This function takes a copy of an image and draws a bounding box
    (polygon) according to the provided `yolo_box` parameter.

    If a `dst_path` is provided, the resulting image will be saved.
    Otherwise, the image will be displayed in a pop up window.

    """
    height, width, _ = image.shape
    if banner_height:
        height = height - banner_height
    else:
        pass

    polygon_1 = [[x * width, y * height] for x, y in yolo_box]
    polygon_1 = np.array(polygon_1, np.int32).reshape((-1, 1, 2))
    is_closed = True

    bgr_tuple = COLOUR_MAPPING[colour]
    text_position = (
        int(TEXT_POSITION_MAPPING[LEGEND_POSITION][colour][0] * width),
        int(TEXT_POSITION_MAPPING[LEGEND_POSITION][colour][1] * height),
    )
    cv2.polylines(image, [polygon_1], is_closed, bgr_tuple, LINE_THICKNESS)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            image, label, text_position, font, 4, bgr_tuple, LINE_THICKNESS, cv2.LINE_AA
        )
    if dst_path:
        cv2.imwrite(str(dst_path), image)
    else:
        cv2.imshow("Un-transformed Bounding Box", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return image


def get_dataframe_from_ai_file(ai_file_path):
    try:
        df = pandas.read_csv(
            filepath_or_buffer=ai_file_path,
            header=None,
            sep=" ",
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            names=[
                "Photo_Name",
                "Class_ID",
                "x1",
                "y1",
                "x2",
                "y2",
                "x3",
                "y3",
                "x4",
                "y4",
                "prob",
            ],
        )
        return True, df
    except Exception:
        df = pandas.read_csv(
            filepath_or_buffer=ai_file_path,
            header=None,
            sep=" ",
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            names=[
                "Photo_Name",
                "Class_ID",
                "x1",
                "y1",
                "x2",
                "y2",
                "x3",
                "y3",
                "x4",
                "y4",
            ],
        )
        return False, df


def save_bounding_boxes_on_images(
    images_root: Path,
    dst_root: Path,
    ai_file_path: Path,
    foot_banner_path: Optional[Path] = None,
):
    """
    Save a copy of all images from images_root to dst_root with bounding boxes applied.
    Only one bounding box is drawn on each image in dst_root.  Filenames in dst_root
    include an index for the defect number.

    If more than one defect was found in an image, there will be multiple corresponding
    images in dst_root.

    """
    if foot_banner_path is not None:
        if not foot_banner_path.exists():
            raise RuntimeError("Cannot find banner file")
        foot_banner = cv2.imread(str(foot_banner_path))
        banner_height, banner_width, _ = foot_banner.shape
        foot_banner = cv2.resize(foot_banner, (1920, int(1920 * banner_height / 1904)))
    else:
        foot_banner = None
        banner_height = 0

    df_has_probabilities, df = get_dataframe_from_ai_file(ai_file_path=ai_file_path)
    images_with_defects = df["Photo_Name"].unique()
    print("\nCount images with defects = ", len(images_with_defects))
    path_adjuster = 0
    while dst_root.exists():
        old_name = dst_root.name.replace(f"_{path_adjuster}", "")
        path_adjuster += 1
        dst_root = dst_root.parent / f"{old_name}_{path_adjuster}"

    dst_root.mkdir(parents=True)

    for img_path in get_all_jpg_recursive(img_root=images_root):
        photo_name = img_path.name
        image_data = df.loc[df["Photo_Name"] == photo_name].reset_index()
        if len(image_data) == 0:
            continue
        orig_image = cv2.imread(str(img_path))
        # orig_height, banner_width, _ = orig_image.shape
        image = cv2.resize(orig_image, (1920, 1080))
        if hasattr(foot_banner, "shape"):
            image = np.concatenate((image, foot_banner), axis=0)
        else:
            pass
        for index, row in image_data.iterrows():
            class_id = str(int(float(row["Class_ID"])))
            series = row[
                [
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "x3",
                    "y3",
                    "x4",
                    "y4",
                ]
            ]
            bounding_box_coords = [
                [series["x1"], series["y1"]],
                [series["x2"], series["y2"]],
                [series["x3"], series["y3"]],
                [series["x4"], series["y4"]],
            ]
            dst_path = dst_root / f"{img_path.stem}{img_path.suffix}"
            if class_id in LABEL_MAPPING:
                label = LABEL_MAPPING[class_id].get("label", "")
                colour = LABEL_MAPPING[class_id].get("colour", GREEN)
            else:
                label = ""
                colour = GREEN

            if df_has_probabilities:
                probability = row["prob"]
                if probability < MARGINAL_PROB_THRESH:
                    colour = GREEN
                else:
                    pass
            else:
                pass
            image = draw_polygon_on_image(
                image=image,
                yolo_box=bounding_box_coords,
                dst_path=dst_path,
                label=label,
                colour=colour,
                banner_height=banner_height,
            )


def _crop_image_for_given_centre(
    img: np.ndarray,
    dim: Tuple[int, int],
    y_centre: float = 0.5,  # for centre_crop y_centre = 0.5
):
    """
    Returns center cropped image unless the centre_crop parameter is set
    to False, in which case cropping removes the image foreground.

    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped

    """
    width, height = img.shape[1], img.shape[0]

    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]

    min_x = int(width / 2.0 - crop_width / 2.0)
    max_x = int(width / 2.0 + crop_width / 2.0)

    min_y = int(height * y_centre - crop_height / 2.0)
    if min_y < 0:
        print("min_y is less than 0")
        min_y = 0
    max_y = min_y + crop_height

    if max_y > height:
        print("max_y is greater than image height")
        max_y = height
        min_y = max_y - crop_height

    crop_img = img[min_y:max_y, min_x:max_x]
    return crop_img


def _scale_image(img: np.ndarray, factor: float):
    """Returns resize image by scale factor.
    This helps to retain resolution ratio while resizing.
    Args:
    img: image to be scaled
    factor: scale factor to resize
    """
    return cv2.resize(img, (int(img.shape[1] * factor), int(img.shape[0] * factor)))


def zoom_image(
    zoom_pcnt: float,
    image: np.ndarray,
    y_centre: float = 0.5,
) -> np.ndarray:
    """
    Return an image that is 'zoomed': same size as the original provided,
    but which has undergone a resize and centre crop.

    Usage::

        `zoom_image(zoom_pcnt=90.0, image=image)`

    will return an image for which the features are approximately 10% larger
    except those features which are near the edge of the image which may
    be partially removed.

    """
    width, height = image.shape[1], image.shape[0]
    factor = 100 / zoom_pcnt
    image = _scale_image(img=image, factor=factor)

    image = _crop_image_for_given_centre(
        img=image, dim=(width, height), y_centre=y_centre
    )
    return image


def make_mp4_movie_from_images_in_dir(
    img_root: Path,
    y_centre: float = 0.5,
    scale: float = 0.35,
    zoom_transition: bool = True,
    fps: float = 1.5,  # frames per second
):
    """
    For each of the images in a directory, 2 additional images that can help
    create a transition effect when creating a video from the still images.

    Applies resize and centre crop to give a progressive zooming into end of
    road at the x = 0.5, y = 0.5.  This works great if the horizon of the image
    is at the centre of the image.

    If zoom_transition is set to True, extra frames are added based on progressive
    zooming. May look good.

    """
    done_once = False
    fps = fps * 52 if zoom_transition else fps
    for img_path in sorted(get_all_jpg_recursive(img_root=img_root)):

        image = cv2.imread(filename=str(img_path))
        image_small = _scale_image(img=image, factor=scale)
        if not done_once:
            frame_size = (image_small.shape[1], image_small.shape[0])
            dst_file = img_root / "an_output_video.mp4"
            out = cv2.VideoWriter(
                str(dst_file), VideoWriter_fourcc(*"mp4v"), fps, frame_size
            )
            done_once = True

        out.write(image=image_small)
        if not zoom_transition:
            continue
        for i in range(20):
            image = zoom_image(zoom_pcnt=99.5, image=image, y_centre=y_centre)
            image_small = _scale_image(img=image, factor=scale)
            out.write(image=image_small)


def test_make_mp4_movie():
    make_mp4_movie_from_images_in_dir(
        img_root=Path("/media/david/Samsung_T8/Hobart_3_roads_risk_labels"),
        y_centre=0.45,
        scale=1.0,
    )
