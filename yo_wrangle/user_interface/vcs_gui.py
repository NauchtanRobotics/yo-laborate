import PySimpleGUI as sg

from yo_wrangle.common import inferred_base_dir
from yo_wrangle.dataset_versioning import commit_and_push

RB_GROUP_1 = "Radio_1"
RB1 = "R1"
RB2 = "R2"
RB3 = "R3"


def backup_train_window():
    message_1 = "BKU: Backup to cloud. "
    message_2 = "EFT: Request 'enhance' training (full). "
    message_3 = "PMT: Submit for 'performance measurement' training. "
    supplementary_text = "Changes primarily to class: "
    base_message = message_1
    message = base_message + supplementary_text
    radio_group = [
                      [sg.Radio(message_1, RB_GROUP_1, enable_events=True, key=RB1, default=True)],
                      [sg.Radio(message_2, RB_GROUP_1, enable_events=True, key=RB2)],
                      [sg.Radio(message_3, RB_GROUP_1, enable_events=True, key=RB3)],
                  ]
    middle_column = [
        [
            sg.Text("Commit Description"),
            sg.In(default_text=message, size=(100, 1), enable_events=True, key="input"),
            sg.Button("Push", key="commit"),
        ],
    ]
    # ----- Full layout -----
    layout = [
        radio_group,
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
        elif event == RB2:
            base_message = message_2
        elif event == RB3:
            base_message = message_3
        elif event == "commit":
            print(message)
            base_dir = inferred_base_dir()
            commit_and_push(
                dataset_label="",  # CLAS-127 How to get dataset label - from Poetry? Do we want to?
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
