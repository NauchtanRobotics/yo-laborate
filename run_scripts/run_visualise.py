from pathlib import Path

from yo_ratchet.yo_wrangle.visualise import save_bounding_boxes_on_images, make_mp4_movie_from_images_in_dir


def test_save_bounding_boxes_on_images():
    images_root = Path("E:\\Aberdeen_Road_FHD")
    dst_root = Path(
        # "D:\\"
        "C:\\CTRC_2021_FHD_polygons_20_pcnt_wedge_horiz_filtered2"
    )  # OneDrive - Shepherd Services\\
    ai_file_path = Path(
        # "C:\\Users\\61419\\OneDrive - Shepherd Services\\Desktop\\Charters_Towers_2021_35_pct_conf_FHD.ai"
        "C:\\defect_detector\\defect_detection\\evaluate\\Charters_Towers_2021_20_pct_conf_FHD.ai"
    )
    foot_banner_path = Path("C:\\Users\\61419\\OneDrive - Shepherd Services\\Desktop\\banner.png")

    save_bounding_boxes_on_images(
        images_root=images_root,
        dst_root=dst_root,
        ai_file_path=ai_file_path,
        foot_banner_path=foot_banner_path,
    )
    make_mp4_movie_from_images_in_dir(
        img_root=dst_root,
        y_centre=0.45,
        scale=1.0,
        zoom_transition=False,
        fps=2.0,
    )


def test_make_mp4_movie():
    make_mp4_movie_from_images_in_dir(
        img_root=Path("/media/david/Samsung_T8/Hobart_3_roads_FHD_flat_rendered4"),
        y_centre=0.45,
        scale=1.0,
        zoom_transition=False,
        fps=2.0,
    )
