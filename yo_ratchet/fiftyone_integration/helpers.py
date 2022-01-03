from fiftyone import DatasetView


def print_dataset_info(dataset: DatasetView):
    print(dataset)
    print("\nBrains Runs:")
    print(dataset.list_brain_runs())
    print("Evaluation Runs:")
    print(dataset.list_evaluations())
