import re

import numpy as np
from datasets import Audio, Dataset, load_dataset

from datasetgenerator.configs import AudioDatasetConfig


def process_same_labels(dataset: Dataset, config: AudioDatasetConfig):
    """
    Process the dataset to ensure all audio samples have the same label.
    This function assumes that the dataset has a 'label' column.
    """
    unique_labels = list(set(dataset[config.target_column]))
    unique_labels = [label.lower() for label in unique_labels]
    dataset = dataset.add_column("all_labels", [unique_labels] * len(dataset))
    return dataset


def remove_special_characters(dataset: Dataset, config: AudioDatasetConfig) -> Dataset:
    """
    Remove special characters from the dataset target_column.
    """

    def clean_text(example):
        text = example[config.target_column]
        cleaned = re.sub(r"[^a-zA-Zа-яА-Я0-9\s]", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        example[config.target_column] = cleaned
        return example

    dataset = dataset.map(clean_text)
    return dataset
