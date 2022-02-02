from pathlib import Path

from yo_ratchet.yo_wrangle.visualise import save_bounding_boxes_on_images, make_mp4_movie_from_images_in_dir


def test_save_bounding_boxes_on_images():
    images_root = Path("/media/david/Samsung_T8/Scenic_Rim_2021_sealed")
    dst_root = Path(
        "/media/david/Samsung_T8/Scenic_Rim_2021_sealed_demo_thresh_2"
        # "C:\\CTRC_2021_FHD_polygons_20_pcnt_wedge_horiz_filtered2"
    )  # OneDrive - Shepherd Services\\
    ai_file_path = Path(
        # "C:\\Users\\61419\\OneDrive - Shepherd Services\\Desktop\\Charters_Towers_2021_35_pct_conf_FHD.ai"
        "/home/david/defect_detection/defect_detection/evaluate/Scenic_Rim_Dec_2021_combined__thresholds_2.ai"
    )
    foot_banner_path = Path("/home/david/Downloads/banner_colour_footer_with_legend.png")
    # Path("C:\\Users\\61419\\OneDrive - Shepherd Services\\Desktop\\banner.png")

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
