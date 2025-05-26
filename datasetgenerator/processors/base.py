import hashlib
import json
import os
import random
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import torch
from tqdm import tqdm

from datasetgenerator.configs import AudioDatasetConfig, DatasetConfig, load_configs
from datasetgenerator.pipelines import (
    ProcessingPipeline,
    normalize_audio,
    process_same_labels,
)
from datasets import Audio, load_dataset

random.seed(42)  # For reproducibility


class BaseProcessor(ABC):
    def __init__(self, dataset_cfgs: list[DatasetConfig]):
        self.dataset_cfgs = dataset_cfgs

    def _normalize_to_list(self, value):
        """Normalize value to always be a list."""
        if value is None:
            return [None]
        elif isinstance(value, list):
            return value
        else:
            return [value]

    @abstractmethod
    def load_datasets(self):
        self.datasets = {}
        """Load datasets based on the configurations provided."""
        pass

    def process(self, pipeline: ProcessingPipeline, out_dataset_name: str = ""):
        """Process the datasets."""
        self.load_datasets()
        print("Datasets loaded and processed.")

        for dataset_name, items in self.datasets.items():
            print(f"Processing dataset: {dataset_name}")
            processed_dataset = pipeline.apply(items["dataset"], items["config"], dataset_name)
            self.datasets[dataset_name]["dataset"] = processed_dataset
            print(f"Dataset {dataset_name} processed successfully.")

        self.save_datasets(out_dataset_name)

    @abstractmethod
    def save_datasets(self, out_dataset_name):
        """Save the processed datasets."""
        pass


class HuggingFaceProcessor(BaseProcessor):
    def _load_single_dataset(self, cfg, subset_name, split):
        """Load a single dataset configuration."""
        try:
            print(f"Loading: {cfg.dataset_name}, subset: {subset_name}, split: {split}")
            dataset_identifier = cfg.dataset_name
            if subset_name is not None:
                dataset_identifier += f"_subset{subset_name}"
            if split is not None:
                dataset_identifier += f"_split{split}"

            kwargs = {}
            if split is not None:
                kwargs["split"] = split
            if subset_name is not None:
                kwargs["name"] = subset_name

            dataset = load_dataset(cfg.dataset_name, **kwargs)
            dataset = dataset.cast_column(cfg.audio_column, Audio(sampling_rate=cfg.sample_rate))

            return dataset_identifier, dataset, cfg

        except Exception as e:
            print(f"Failed to load {cfg.dataset_name} (subset: {subset_name}, split: {split}): {e}")
            return None, None

    def load_datasets(self):
        self.datasets = {}
        for cfg in self.dataset_cfgs:
            subset_names = self._normalize_to_list(getattr(cfg, "subset_name", None))
            splits = self._normalize_to_list(getattr(cfg, "split", None))
            for subset_name, split in product(subset_names, splits):
                dataset_identifier, dataset, cfg = self._load_single_dataset(cfg, subset_name, split)
                if dataset:
                    self.datasets[dataset_identifier] = {"dataset": dataset, "config": cfg}
        print(self.datasets)

    def save_audio_features(self, audio_data, audio_dir):
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data)

        audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()
        audio_filename = f"{audio_hash}.pt"
        audio_path = os.path.join(audio_dir, audio_filename)
        torch.save(audio_data, audio_path)
        return audio_path

    def save_datasets(self, out_dataset_name):
        """Save the processed datasets to disk."""
        save_path = os.path.join("datasets", out_dataset_name)
        audio_dir = os.path.join(save_path, "audio_features")
        output_dataset_file = out_dataset_name if out_dataset_name.endswith(".json") else f"{out_dataset_name}.json"

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(audio_dir, exist_ok=True)

        dataset_list = []
        for dataset_identifier, items in self.datasets.items():
            dataset = items["dataset"]
            config = items["config"]
            for _, example in enumerate(tqdm(dataset, desc=f"Creating: {out_dataset_name}", total=len(dataset))):
                audio_data = example[f"{config.audio_column}_features"]
                audio_path = self.save_audio_features(audio_data, audio_dir)
                label = example[config.target_column].lower()
                all_labels = example["all_labels"]
                random.shuffle(all_labels)
                row = {
                    "source_dataset": dataset_identifier,
                    "audio_features_path": audio_path,
                    "all_labels": all_labels,
                    "true_labels": [label],
                }

                dataset_list.append(row)
        random.shuffle(dataset_list)
        with open(os.path.join(save_path, output_dataset_file), "w") as f:
            json.dump(dataset_list, f, indent=4)


if __name__ == "__main__":
    pipeline = ProcessingPipeline()
    pipeline.add_step("normalize_audio", normalize_audio, batch_size=2)
    pipeline.add_step("process_same_labels", process_same_labels)

    # configs = load_configs("datasetgenerator/configs/configs/gliclass-audio.json", "audio")
    configs = load_configs("./datasetgenerator/configs/configs/test.json", "audio")
    processor = HuggingFaceProcessor(configs)
    processor.process(pipeline, out_dataset_name="gliclass-audio-datset")
