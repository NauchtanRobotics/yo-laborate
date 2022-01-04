import PySimpleGUI as sg
import fiftyone as fo
from pathlib import Path
from typing import List

from yo_ratchet.fiftyone_integration import find_errors
from yo_ratchet.yo_wrangle.common import inferred_base_dir, get_classes_list

EXPORT_FOLDER_NAME = ".export"
DATASET_KEY = "-DATASET-"
CLASS_KEY = "-CLASS-"
METHOD_KEY = "-METHOD-"
YES_CONFIRMATION = "Yes"


def list_available_fo_datasets(base_dir: Path) -> List[str]:
    """
    Lists the union of installed fo datasets (in fo database) plus all of the dataset
    available to import from <base_dir>/.exports/*.

    """
    installed_dataset_names = fo.list_datasets()
    exports_available = [folder.name for folder in (base_dir / EXPORT_FOLDER_NAME).iterdir()]
    dataset_names = sorted(list(set(installed_dataset_names + exports_available)), reverse=True)
    return dataset_names


def launch_find_errors_config_window(base_dir: Path = None):
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
    available_datasets = list_available_fo_datasets(base_dir=base_dir)
    middle_col = [
            # DS: Name DROPDOWN for Dataset Names - or KISS: Do we want to risk the wrong DS being opened?
            [sg.Text("Dataset")],
            [
                sg.Listbox(
                    values=available_datasets,
                    select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
                    size=(30, 3),
                    enable_events=False,
                    visible=True,
                    key=DATASET_KEY,
                )
            ],
            # TAG: DROPDOWN for error type tag
            [sg.Text("Method")],
            [
                sg.Listbox(
                    values=["mistakenness", "eval_fp", "eval_fn"],
                    select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
                    size=(30, 3),
                    enable_events=False,
                    visible=True,
                    key=METHOD_KEY,
                )
            ],
            # CLASS: DROPDOWN SELECT class FROM CLASS LIST
            [sg.Text("Class")],
            [
                sg.Listbox(
                    values=class_names_list,
                    select_mode=sg.LISTBOX_SELECT_MODE_SINGLE,
                    size=(30, 6),
                    enable_events=True,
                    visible=True,
                    change_submits=True,
                    key=CLASS_KEY,
                )
            ],
            # LIMIT: INTEGER INPUT FIELD
            # [sg.Text("#Results")],

            [sg.Button("GO", key="-GO-")],
    ]
    # ----- Full layout -----
    layout = [
        [
            sg.Column(middle_col),
        ],
    ]
    window = sg.Window(
        "Find Errors in Dataset",
        layout,
        modal=True,
        margins=(50, 20),
        finalize=True,
    )
    selected_method = window[METHOD_KEY]
    selected_method.update(set_to_index=[0], scroll_to_index=0)
    selected_dataset_name = window[DATASET_KEY]
    selected_dataset_name.update(set_to_index=[0], scroll_to_index=0)
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "-GO-":
            if len(values[CLASS_KEY]) == 0:
                sg.popup("Please select a class")
                continue
            else:
                selected_class = values[CLASS_KEY][0]

            if len(values[METHOD_KEY]) == 0:
                sg.popup("Please select an error finding Method.")
                continue
            else:
                selected_method = values[METHOD_KEY][0]

            if len(values[DATASET_KEY]) == 0:
                sg.popup("Please selected a Dataset.")
                continue
            else:
                selected_dataset_name = values[DATASET_KEY][0]

            if selected_dataset_name not in fo.list_datasets():
                ret = sg.popup_yes_no(f"Dataset {selected_dataset_name} not installed. Import now?")
                if ret == YES_CONFIRMATION:
                    path_to_import = base_dir / EXPORT_FOLDER_NAME / selected_dataset_name
                    labels_path = str(path_to_import)
                    data_path = str(base_dir)
                    dataset = fo.Dataset.from_dir(
                        dataset_dir=labels_path,
                        rel_dir=data_path,
                        dataset_type=fo.types.FiftyOneDataset,
                        name=selected_dataset_name,
                    )
                    dataset.persistent = True
                    dataset.save()
                    sg.popup(f"This may take 5-10 min. \nGo enjoy a coffee!\nImporting: {str(path_to_import)}")
                continue
            else:
                pass  # selected dataset should be fine to be opened by find_errors()
            find_errors(
                dataset_label=selected_dataset_name,
                class_names=class_names_list,
                tag=selected_method,
                limit=86,  # make this an input field. Cannot handle >= 90 images in Windows for some reason
                processed=True,
                reverse=True,
                label_filter=selected_class,
                base_dir=base_dir,
            )
        else:
            print("Unknown event: ", event)

    window.close()
    return ""


if __name__ == "__main__":
    launch_find_errors_config_window()
