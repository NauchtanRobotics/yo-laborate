from pathlib import Path

from yo_wrangle.workflow import (
    run_prepare_dataset_and_train,
    set_globals,
    run_find_errors,
)
import wrangling_example as dataset_workbook


def test_prepare_dataset_and_train():
    set_globals(base_dir=Path(__file__).parent, workbook_ptr=dataset_workbook)
    run_prepare_dataset_and_train()


def test_find_errors():
    set_globals(base_dir=Path(__file__).parent, workbook_ptr=dataset_workbook)
    run_find_errors(
        tag="mistakenness",
        label_filter="WS",
        limit=64,
    )


def test_export():
    import fiftyone as fo

    dataset = fo.load_dataset("v9b")
    dataset.export(
        export_dir="./.export",
        dataset_type=fo.types.FiftyOneDataset,
        export_media=False,
    )


def test_import():
    # from fiftyone.utils.data.exporters import FiftyOneDatasetExporter # FiftyOneDatasetImporter as Importer
    name = "dsXXX"
    import fiftyone as fo

    if name in fo.list_datasets():
        fo.delete_dataset(name=name)
    else:
        pass
    dataset = fo.Dataset.from_dir(
        dataset_dir="./.export",
        dataset_type=fo.types.FiftyOneDataset,
        name=name,
    )
    # fo.launch_app(dataset)
    set_globals(base_dir=Path(__file__).parent, workbook_ptr=dataset_workbook)
    run_find_errors(
        tag="mistakenness",
        label_filter="WS",
        limit=64,
        dataset_label=name,
    )
