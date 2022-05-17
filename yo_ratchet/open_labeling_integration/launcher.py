import sys
import subprocess
from pathlib import Path

from typing import List
from open_labeling.launcher import POETRY_APP, SYS_STDOUT, SYS_STDERR


def edit_labels(filenames: List[str], class_names: List[str], base_dir: Path):
    """Opens OpenLabeling with this list of images filenames found in root_folder
    as per provided parameters.

    Reduces the effort of manually checking images, identifying possible labelling errors
    then having to manually search for these and edit in another application.

    """
    assert base_dir.exists(), f"base_dir does not exist: {str(base_dir)}"
    cmd = [
        str(POETRY_APP),
        "env",
        "info",
        "--path",
    ]
    try:
        res = subprocess.check_output(cmd, cwd=str(base_dir))
    except subprocess.CalledProcessError as error:
        message = f"{error} | base_dir = {str(base_dir)}"
        raise RuntimeError(message)
    if res is None:
        raise RuntimeError(f"Poetry env not installed. Res = {res}")

    open_labeling_app = res.decode("utf8").splitlines()[0]
    open_labeling_app = Path(open_labeling_app)
    if sys.platform == "win32":
        open_labeling_app = (
            open_labeling_app / "Lib" / "site-packages" / "open_labeling" / "run_app.py"
        ).resolve()
    else:
        open_labeling_app = (
            open_labeling_app
            / "lib"
            / "python3.8"
            / "site-packages"
            / "open_labeling"
            / "run_app.py"
        ).resolve()
    assert open_labeling_app.exists(), f"Path does not exist: {str(open_labeling_app)}"
    print(str(open_labeling_app))

    cmd = [
        str(POETRY_APP),
        "run",
        "python",
        f"{str(open_labeling_app)}",
        "-c",
        *class_names,
        "--files-list",
        *filenames,
    ]
    subprocess.run(
        args=cmd, stdout=SYS_STDOUT, stderr=SYS_STDERR, check=True, cwd=str(base_dir)
    )
