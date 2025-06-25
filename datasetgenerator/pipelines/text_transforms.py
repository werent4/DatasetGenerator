import re

import numpy as np
from datasets import Audio, Dataset, load_dataset

from datasetgenerator.configs import AudioDatasetConfig


def process_same_labels(dataset: Dataset, config: AudioDatasetConfig):
    """
    Process the dataset to ensure all audio samples have the same label.
    This function assumes that the dataset has a 'label' column.
    """
    if isinstance(dataset[config.target_column][0], str):
        unique_labels = list(set(dataset[config.target_column]))
    if isinstance(dataset[config.target_column][0], list):
        all_labels = []
        for labels_list in dataset[config.target_column]:
            all_labels.extend(labels_list)
        
        unique_labels = list(set(all_labels))

    unique_labels = [label.lower() for label in unique_labels]
    dataset = dataset.add_column("all_labels", [unique_labels] * len(dataset))
    return dataset


def remove_special_characters(dataset: Dataset, config: AudioDatasetConfig) -> Dataset:
    """
    Remove special characters from the dataset target_column.
    """
    pattern = r"[^a-zA-Zа-яА-Я0-9\s]"
    def clean_text(example):
        text = example[config.target_column]
        if isinstance(text, str):
            cleaned = re.sub(pattern, " ", text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            example[config.target_column] = cleaned
        elif isinstance(text, list):
            cleaned_list = []
            for item in text:
                if isinstance(item, str):
                    cleaned = re.sub(pattern, " ", item)
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()
                    cleaned_list.append(cleaned)
                else:
                    cleaned_list.append(item) 
            example[config.target_column] = cleaned_list
        return example

    dataset = dataset.map(clean_text)
    return dataset
