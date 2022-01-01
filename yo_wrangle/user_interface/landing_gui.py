import PySimpleGUI as sg

from yo_wrangle.common import get_classes_list, inferred_base_dir
from yo_wrangle.user_interface.vcs_gui import backup_train_window
from open_labeling.launcher import main as open_labeling_launcher

BACKUP_TRAIN_VCS = "Backup/Train"
LABEL_FOLDER = "Label Folder"
EXPLORE_DS = "Explore Dataset"
FIND_ERRORS = "Find/Edit Errors"


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
            base_dir = inferred_base_dir()
            class_labels_list = get_classes_list(base_dir)

            class Args:
                class_list = *class_labels_list,

            open_labeling_launcher(args=Args())
        elif event == FIND_ERRORS:
            pass

    window.close()


if __name__ == "__main__":
    main()


# def test_inferred_base_dir():
#     base_dir = inferred_base_dir()
#     print(str(base_dir))

