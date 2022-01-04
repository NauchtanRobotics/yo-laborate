import subprocess
from pathlib import Path
from typing import Optional

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


def get_path_for_best_pretrained_model(base_dir: Path):
    """
    Provides the path to a pretrained model that can provide initial
    weights to commence model fine-tuning training.

    Weights from an incremental model for your classification problem
    can be used as a starting point for training you next model, or
    weights from a pre-trained model for a different classification problem.

    The starting point which will lead to convergence in the smallest amount
    of epochs will depend on a number of factors... so use judgement
    when using this function.

    More work is required in this function to ensure that it meets the
    'best' criteria.

    """
    tag_text = get_highest_tag_text(base_dir=base_dir)
    if tag_text is not None:
        weights_path = get_model_path_corresponding_to_tag(
            base_dir=base_dir, tag_text=tag_text
        )
    else:
        (_, _, _, weights_path, _, _, _) = get_config_items(base_dir=base_dir)
    return weights_path


def test_get_highest_tag():
    base_dir = Path("/home/david/RACAS/sealed_roads_dataset")
    tag = get_highest_tag_text(base_dir=base_dir)
    assert isinstance(tag, str) or tag is None


def test_get_path_for_best_pretrained_model():
    base_dir = Path("/home/david/RACAS/sealed_roads_dataset")
    weights_path = get_path_for_best_pretrained_model(base_dir=base_dir)
    print(f"\nPath: {str(weights_path)}")
