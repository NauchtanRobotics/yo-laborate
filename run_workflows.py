from pathlib import Path
from yo_wrangle.workflow import run_prepare_dataset_and_train, set_globals
import wrangling_example as dataset_workbook


def test_prepare_dataset_and_train():
    set_globals(base_dir=Path(__file__).parent, workbook_ptr=dataset_workbook)
    run_prepare_dataset_and_train()
