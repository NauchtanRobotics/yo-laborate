import os
os.environ["FIFTYONE_DISABLE_SERVICES"] = "1"

import PySimpleGUI as sg
from pathlib import Path

from fiftyone_integration import edit_labels
from yo_ratchet.yo_filter.unsupervised import find_n_most_distant_outliers_in_batch
from yo_ratchet.yo_wrangle.common import inferred_base_dir, get_classes_list, get_yolo_detect_paths, \
    get_label_to_id_map

MAX_NUM_OUTLIERS = "Max Num Outliers"

N_ = "-N-"
CLASS_ID_ = "-CLASS_ID-"
TEST_ = "-TEST-"
TRAIN_ = "-TRAIN-"
CLASS_ = "-CLASS-"
GO_ = "-GO-"


def launch_find_outliers_config_window(base_dir: Path = None):
    """
    # DROPDOWN FROM REV SORTED LIST FO.list_datasets()
    # ..NEED BUTTON TO INSTALL ANY NEW DATASET from base_dir/.exports/* or MODAL if ds not in fo.list_datasets
    # DROPDOWN for error type tag (HARD CODED CHOICES)
    # INTEGER INPUT FIELD
    # DROPDOWN SELECT FROM CLASS LIST
    """
    if base_dir is None:
        base_dir = inferred_base_dir()
        print("Inferred base dir")
    else:
        print("Received base dir: ", str(base_dir))
    class_names_list = get_classes_list(base_dir=base_dir)
    _, yolo_root = get_yolo_detect_paths(base_dir=base_dir)
    label_to_id_dict = get_label_to_id_map(base_dir=base_dir)
    datasets_root = yolo_root / "datasets"

    file_list_column = [
        [
            sg.Text(MAX_NUM_OUTLIERS, size=(17, 1)),
            sg.In(default_text=5, size=(40, 1), enable_events=True, key=N_),
        ],
        [
            sg.Text("Test Data", size=(17, 1)),
            sg.In(size=(40, 1), enable_events=True, key=TEST_),
            sg.FolderBrowse(initial_folder=str(base_dir)),
        ],
        [
            sg.Text("Training Data", size=(17, 1)),
            sg.In(size=(40, 1), enable_events=True, key=TRAIN_),
            sg.FolderBrowse(initial_folder=str(datasets_root)),
        ],
        [
            sg.Text("Class", size=(17, 1)),
            sg.Listbox(
                values=class_names_list,
                select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
                size=(38, 12),
                enable_events=True,
                visible=True,
                change_submits=True,
                key=CLASS_,
            ),
            sg.Button("GO", key=GO_, size=(7, 1)),
        ],
    ]

    # ----- Full layout -----
    layout = [
        [
            sg.Column(file_list_column),
        ]
    ]

    window = sg.Window(title="OpenLabeling Launcher", layout=layout, margins=(80, 15))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == GO_:
            if len(values[N_]) == 0:
                sg.popup("Please select maximum number of outliers")
                continue
            else:
                try:
                    n_outliers = int(values[N_])
                except:
                    sg.popup("Please enter an integer value for '%s'" % MAX_NUM_OUTLIERS)
                    continue

            if len(values[TRAIN_]) == 0:
                sg.popup("Please select a folder for training data")
                continue
            else:
                train_data = Path(values[TRAIN_])

            if len(values[TEST_]) == 0:
                sg.popup("Please select a folder for test data")
                continue
            else:
                test_data = Path(values[TEST_])

            if len(values[CLASS_]) == 0:
                sg.popup("Please select a class")
                continue
            else:
                selected_class_label = values[CLASS_][0]

            class_id = label_to_id_dict.get(selected_class_label, None)
            try:
                class_id = int(class_id)
            except:
                if class_id is None:
                    sg.popup(f"Could not find class '{selected_class_label}' in classes.json")
                else:
                    sg.popup(f"Could not convert class_id '{class_id}' to int")
                continue
            sg.popup("This will probably take a few minutes. Time for a coffee!")
            image_names = find_n_most_distant_outliers_in_batch(
                train_data=train_data,
                test_data=test_data,
                class_id=class_id,
                layer_number=80,  # ADD A DROPDOWN WITH SEVERAL WORKABLE OPTIONS
                n_outliers=n_outliers,
            )
            image_paths = [test_data / image_name for image_name in image_names]
            image_paths = [str(image_path) for image_path in image_paths]
            print(image_paths)
            edit_labels(filenames=image_paths, class_names=class_names_list, base_dir=base_dir)
        else:
            print("Event: ", event, "; Selected: ", values[event])

    window.close()
    return ""


if __name__ == "__main__":
    launch_find_outliers_config_window()
