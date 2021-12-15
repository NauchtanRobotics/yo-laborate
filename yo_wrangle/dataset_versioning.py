import subprocess
import sys
from pathlib import Path
from typing import Optional

from yo_wrangle.common import get_version_control_config


def commit_and_push(
    dataset_label: str,
    base_dir: Path,
    description: str = "changes and pre-artifacts",
    remote_name: Optional[str] = None,
    remote_branch: Optional[str] = None,
):
    """
    Commit all changes then push to remote, via three commands:
    * git add -A
    * git commit -m'<message>'
    * git push origin <remote_branch>

    """
    git_exe_path, config_remote_name, config_branch_name = get_version_control_config(
        base_dir
    )
    if remote_name is None:
        remote_name = config_remote_name
    if remote_branch is None:
        remote_branch = config_branch_name

    try:
        cmd = [
            git_exe_path,
            "add",
            ".",
        ]
        subprocess.check_call(
            args=cmd,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            cwd=str(base_dir),
        )
        cmd = [git_exe_path, "commit", "-m", f"{dataset_label} {description}"]
        subprocess.check_call(
            args=cmd,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            cwd=str(base_dir),
        )
        cmd = [
            git_exe_path,
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
        description="Formatting",
    )
