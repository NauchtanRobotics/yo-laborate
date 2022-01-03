import subprocess
from pathlib import Path
from typing import Optional

from yo_ratchet.yo_wrangle.common import get_version_control_config


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
        subprocess.run(
            args=cmd,
            cwd=str(base_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        full_message = f"{dataset_label} {description}" if len(dataset_label) > 0 else description
        cmd = [git_exe_path, "commit", "-m", full_message]
        subprocess.run(
            args=cmd,
            cwd=str(base_dir),
            check=True,
        )
        cmd = [
            git_exe_path,
            "push",
            f"{remote_name}",
            f"{remote_branch}",
        ]
        subprocess.run(
            args=cmd,
            cwd=str(base_dir),
            check=True,
        )
    except subprocess.CalledProcessError:
        print("Git Error. Probably no changes to commit. Continuing...")


def test_git_commit_and_push():
    commit_and_push(
        dataset_label="",
        base_dir=Path(__file__).parents[2],
        description="TEST",
        remote_branch="CLAS-127"
    )
