import PySimpleGUI as sg

from yo_ratchet.dataset_versioning.version import (
    get_dataset_label_from_version,
    bump_patch,
    bump_minor_and_remove_patch,
)
from yo_ratchet.yo_wrangle.common import inferred_base_dir
from yo_ratchet.dataset_versioning import commit_and_push

PATCH = "-PATCH-"

RB_GROUP_1 = "Radio_1"
RB1 = "R1"
RB2 = "R2"
RB3 = "R3"


def backup_train_window():
    message_1 = "BKU: Backup to cloud. "
    message_2 = "TI: Backup and request incremental training (1:5 left out). "
    message_3 = "TD: Backup and request 1:1 double training (not recommended). "
    supplementary_text = "Changes primarily to class: "
    base_message = message_1
    message = base_message + supplementary_text
    radio_group = [
        [sg.Radio(message_1, RB_GROUP_1, enable_events=True, key=RB1, default=True)],
        [sg.Radio(message_2, RB_GROUP_1, enable_events=True, key=RB2)],
        [sg.Radio(message_3, RB_GROUP_1, enable_events=True, key=RB3)],
    ]
    check_boxes = [
        [sg.Checkbox("Only increment the version patch.", key=PATCH, visible=True, disabled=True)],
    ]
    middle_column = [
        [
            sg.Text("Commit Description"),
            sg.In(default_text=message, size=(100, 1), enable_events=True, key="input"),
        ],
        [sg.Button("Push", key="commit"),]
    ]
    # ----- Full layout -----
    layout = [
        radio_group,
        check_boxes,
        [
            sg.Column(middle_column),
        ],
    ]
    window = sg.Window("Dataset Version Control", layout, modal=True, margins=(50, 20))
    nominal_class = ""
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "input":
            assembled_message = values["input"]
            nominal_class = assembled_message.split(":")[-1].strip()
        elif event == RB1:
            base_message = message_1
            window[PATCH].Update(disabled=True)
        elif event == RB2:
            base_message = message_2
            window[PATCH].Update(disabled=False)
        elif event == RB3:
            base_message = message_3
            window[PATCH].Update(disabled=False)
        elif event == "commit":
            print(message)
            base_dir = inferred_base_dir()
            if values[RB2] or values[RB3]:  # Training was requested
                if values[PATCH] is True:
                    bump_patch(base_dir=base_dir)
                else:
                    bump_minor_and_remove_patch(base_dir=base_dir)
            else:
                pass  # Do not bump version if it is only a "save-point"
            commit_and_push(
                dataset_label=get_dataset_label_from_version(base_dir=base_dir),
                base_dir=base_dir,
                description=message,
            )
            break
        else:
            print("Unknown event: ", event)
        message = base_message + supplementary_text + nominal_class
        window["input"].update(value=message)
    window.close()
    return message


if __name__ == "__main__":
    backup_train_window()
