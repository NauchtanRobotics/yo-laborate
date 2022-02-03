import subprocess
from pathlib import Path
from typing import Optional, Tuple

from dataset_versioning.version import is_minor_absent
from yo_ratchet.yo_wrangle.common import get_config_items

GET_TAG_HASH_CMD = ["git", "rev-list", "--tags", "--max-count=1"]
GET_TAG_TEXT_CMD = ["git", "describe", "--tags"]


def get_highest_tag_text(base_dir: Path) -> Optional[str]:
    highest_tag_hash = subprocess.check_output(GET_TAG_HASH_CMD, cwd=str(base_dir))
    highest_tag_hash = highest_tag_hash.decode("utf-8")
    if highest_tag_hash == "":
        return None
    highest_tag_hash = highest_tag_hash.splitlines()[0]

    get_tag_text_cmd = GET_TAG_TEXT_CMD + [highest_tag_hash]
    highest_tag_text = subprocess.check_output(get_tag_text_cmd, cwd=str(base_dir))
    highest_tag_text = highest_tag_text.decode("utf-8").splitlines()[0]
    return highest_tag_text


def get_model_path_corresponding_to_tag(base_dir: Path, tag_text: str) -> Path:
    (_, yolo_base_dir, _, _, _, _, _) = get_config_items(base_dir=base_dir)
    model_path = (
        Path(yolo_base_dir) / "runs" / "train" / tag_text / "weights" / "best.pt"
    )
    assert model_path.exists(), f"Path does not exist: {str(model_path)}"
    return model_path


def get_path_for_best_pretrained_model(base_dir: Path) -> Tuple[Path, bool]:
    """
    Provides the path to a pretrained model that can provide initial
    weights to commence model training, and a boolean to indicate whether
    fine-tuning of a customised model is required (requiring a smaller initial
    learning rate) or the training will be based on generic pretrained model
    (requiring the default initial learning rate).

    Weights from an incremental model for your classification problem
    can be used as a starting point for training you next model, or
    weights from a pre-trained model for a different classification problem.

    The starting point which will lead to convergence in the smallest amount
    of epochs will depend on a number of factors... so use judgement
    when using this function.

    A data analyst/scientist can trigger full retraining from epoch 0 by
    removing the patch and minor part of the version in pyproject.toml file.
    This is advisable upon changes to the classes list / or major shifts in
    the boundaries there-between as fine tuning would be insufficient in
    such a context.

    """
    tag_text = get_highest_tag_text(base_dir=base_dir)
    if is_minor_absent(base_dir=base_dir) or tag_text is None:
        (_, _, _, weights_path, _, _, _) = get_config_items(base_dir=base_dir)
        fine_tune = False
    else:
        weights_path = get_model_path_corresponding_to_tag(
            base_dir=base_dir, tag_text=tag_text
        )
        fine_tune = True
    return weights_path, fine_tune


def test_get_highest_tag():
    base_dir = Path("/home/david/RACAS/sealed_roads_dataset")
    tag = get_highest_tag_text(base_dir=base_dir)
    assert isinstance(tag, str) or tag is None


def test_get_path_for_best_pretrained_model():
    base_dir = Path("/home/david/RACAS/sealed_roads_dataset")
    weights_path, fine_tune = get_path_for_best_pretrained_model(base_dir=base_dir)
    print(f"\nPath: {str(weights_path)}")
    print(f"Mode is pretrained / only requires fine_tune: {fine_tune}")
