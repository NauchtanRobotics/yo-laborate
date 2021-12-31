import PySimpleGUI as sg


BACKUP_TRAIN_VCS = "Backup/Train"
LABEL_FOLDER = "Label Folder"
EXPLORE_DS = "Explore Dataset"
FIND_ERRORS = "Find/Edit Errors"


def backup_train_window():
    message_1 = "Backup to cloud. "
    message_2 = "EFT: Request 'enhance' training (full). "
    message_3 = "PMT: Submit for 'performance measurement' training. "
    supplementary_text = "Changes primarily to class: "
    message = message_1
    radio_group = [
                      [sg.Radio(message_1, 1, enable_events=True, key='R1', default=True)],
                      [sg.Radio(message_2, 1, enable_events=True, key='R2')],
                      [sg.Radio(message_3, 1, enable_events=True, key='R3')],
                  ]
    middle_column = [
        [
            sg.Text("Commit Description"),
            sg.In(default_text=message_1+supplementary_text, size=(100, 1), enable_events=True, key="input"),
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
            message = values["input"]
            nominal_class = message.split(":")[-1].strip()
        elif event == "R1":
            message = message_1 + supplementary_text + nominal_class
        elif event == "R2":
            message = message_2 + supplementary_text + nominal_class
        elif event == "R3":
            message = message_3 + supplementary_text + nominal_class
        elif event == "commit":
            print(message)
            break
        else:
            print("Unknown event: ", event)
        window["input"].update(value=message)
    window.close()
    return message


def main():
    sg.ChangeLookAndFeel('LightGreen')

    # ------ Menu Definition ------ #
    menu_def = [
        ["Actions", [LABEL_FOLDER, BACKUP_TRAIN_VCS, EXPLORE_DS, FIND_ERRORS, "---", "Exit"]],
        ['Help', 'About...'],
    ]
    middle_column = [
        [sg.Text("Actions Log:")],
        [sg.Output(size=(60, 25), key="console",)]
    ]
    col_element = sg.Column(middle_column,)

    # ----- Full layout -----
    layout = [
        [sg.Menu(menu_def, )],
        [col_element],
    ]
    window = sg.Window(
        "YO-Laborate",
        layout,
        margins=(90, 20),
        resizable=True,
        finalize=True,
        default_element_size=(12, 1),
        auto_size_text=False,
        auto_size_buttons=False,
        default_button_element_size=(12, 1),
    )

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event is not None:
            print(event)
            window.refresh()

        if event == BACKUP_TRAIN_VCS:
            backup_train_window()
        elif event == LABEL_FOLDER:
            pass
        elif event == FIND_ERRORS:
            pass

    window.close()


if __name__ == "__main__":
    main()
