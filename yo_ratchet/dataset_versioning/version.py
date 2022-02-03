import subprocess
from pathlib import Path

from open_labeling.launcher import POETRY_APP

VERSION_CMD = [str(POETRY_APP), "version"]
PATCH = "patch"
MINOR = "minor"
MAJOR = "major"


def bump_patch(base_dir: Path):
    cmd = VERSION_CMD + [PATCH]
    subprocess.run(args=cmd, cwd=str(base_dir))


def bump_minor(base_dir: Path):
    cmd = VERSION_CMD + [MINOR]
    subprocess.check_output(args=cmd, cwd=str(base_dir))


def bump_minor_and_remove_patch(base_dir: Path):
    bump_minor(base_dir)
    version = get_version(base_dir=base_dir)
    version = version.split(".")
    version = ".".join(version[:2])  # Removes the patch in third position
    cmd = VERSION_CMD + [version]
    subprocess.check_output(args=cmd, cwd=str(base_dir))


def bump_major(base_dir: Path):
    cmd = VERSION_CMD + [MAJOR]
    subprocess.run(args=cmd, cwd=str(base_dir))


def get_version(base_dir: Path) -> str:
    cmd = VERSION_CMD + ["-sn"]
    res = subprocess.check_output(args=cmd, cwd=str(base_dir))
    res = res.decode("utf-8")
    res = res.splitlines()[0]
    return res


def get_minor(base_dir: Path) -> int:
    minor = get_version(base_dir=base_dir).split(".")[1]
    return int(minor)


def get_patch(base_dir: Path) -> int:
    patch = get_version(base_dir=base_dir).split(".")[2]
    return int(patch)


def is_major_version(base_dir: Path) -> bool:
    minor = get_minor(base_dir=base_dir)
    patch = get_patch(base_dir=base_dir)
    if minor == 0 and patch == 0:
        return True
    else:
        return False


def is_minor_absent(base_dir: Path) -> bool:
    version_split = get_version(base_dir=base_dir).split(".")
    if len(version_split) > 1:
        return False
    else:
        return True


def get_repository_abbreviation(base_dir: Path) -> str:
    name = base_dir.name
    if "-" in name:
        name_parts = name.split("-")
    else:
        name_parts = name.split("_")

    abbreviation = [part[0] for part in name_parts]
    abbreviation = "".join(abbreviation)
    return abbreviation


def get_dataset_label_from_version(base_dir: Path) -> str:
    dataset_label = get_repository_abbreviation(base_dir=base_dir)
    dataset_label += get_version(base_dir=base_dir)
    return dataset_label


"""################## TESTING #########################"""

BASE_DIR = Path(__file__).parents[2]


def test_bump_minor_and_remove_patch():
    bump_minor_and_remove_patch(base_dir=BASE_DIR)


def test_get_repository_abbreviation():
    abbrev = get_repository_abbreviation(base_dir=BASE_DIR)
    assert abbrev == "yw"


def test_get_dataset_label_from_version():
    print(get_dataset_label_from_version(base_dir=BASE_DIR))


def test_get_version():
    res = get_version(base_dir=BASE_DIR)
    print(res)


def test_bump_minor():
    bump_minor(base_dir=BASE_DIR)
    res = get_version(base_dir=BASE_DIR)
    print(res)


def test_bump_major():
    bump_major(base_dir=BASE_DIR)
    res = get_version(base_dir=BASE_DIR)
    print(res)


def test_is_major():
    assert is_major_version(base_dir=BASE_DIR) is False
