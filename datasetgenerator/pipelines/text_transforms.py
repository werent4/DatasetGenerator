import numpy as np

from datasetgenerator.configs import AudioDatasetConfig
from datasets import Audio, Dataset, load_dataset


def process_same_labels(dataset: Dataset, config: AudioDatasetConfig):
    """
    Process the dataset to ensure all audio samples have the same label.
    This function assumes that the dataset has a 'label' column.
    """
    unique_labels = list(set(dataset[config.target_column]))
    unique_labels = [label.lower() for label in unique_labels]
    dataset = dataset.add_column("all_labels", [unique_labels] * len(dataset))
    return dataset
