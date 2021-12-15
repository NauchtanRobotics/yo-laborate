import subprocess
import sys
from pathlib import Path

GIT_EXE = "/usr/bin/git"


def commit_and_push(
    dataset_label: str,
    base_dir: Path,
    description: str = "changes and pre-artifacts",
    remote_name: str = "origin",
    remote_branch: str = "master",
):
    """
    Commit all changes then push to remote, via three commands:
    * git add -A
    * git commit -m'<message>'
    * git push origin <remote_branch>

    """
    try:
        cmd = [
            GIT_EXE,
            "add",
            ".",
        ]
        subprocess.check_call(
            args=cmd,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            cwd=str(base_dir),
        )
        cmd = [
            GIT_EXE,
            "commit",
            "-m",
            f"{dataset_label} {description}"
        ]
        subprocess.check_call(
            args=cmd,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            cwd=str(base_dir),
        )
        cmd = [
            GIT_EXE,
            "push",
            f"{remote_name}",
            f"{remote_branch}",
        ]
        subprocess.check_call(
            args=cmd,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            cwd=str(base_dir),
        )
    except subprocess.CalledProcessError:
        print("Git error (probably no changes to commit), continuing...")


def test_git_commit_and_push():
    commit_and_push(
        dataset_label="CLAS-119",
        base_dir=Path(__file__).parents[1],
        description=" Auto git commit within workflow."
    )
