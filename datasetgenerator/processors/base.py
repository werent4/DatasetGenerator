import hashlib
import json
import os
import random
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

from datasetgenerator.configs import AudioDatasetConfig, DatasetConfig, load_configs
from datasetgenerator.pipelines import ProcessingPipeline

random.seed(42)  # For reproducibility


class BaseProcessor(ABC):
    def __init__(self, dataset_cfgs: list[DatasetConfig], pipelines: list[ProcessingPipeline] = None):
        if not dataset_cfgs:
            raise ValueError("dataset_cfgs must be provided and cannot be empty.")
        if pipelines is None:
            raise ValueError("pipelines must be provided and cannot be None.")

        self.dataset_cfgs = dataset_cfgs
        self.pipelines = pipelines

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

    def process(self, out_dataset_name: str = ""):
        """Process the datasets."""
        self.load_datasets()
        print("Datasets loaded and processed.")

        for pipeline_name, pipeline  in self.pipelines:
            print(f"Applying pipeline: {pipeline.__class__.__name__}")
            pipeline.list_steps()

            for dataset_name, items in self.datasets.items():
                config = items["config"]
                dataset = items["dataset"]

                if config.pipelines_to_skip is not None and pipeline_name in config.pipelines_to_skip:
                    print(f"Skipping pipeline {pipeline.__class__.__name__} for dataset {dataset_name} due to configuration.")  
                    continue

                print(f"Processing dataset: {dataset_name}")
                processed_dataset = pipeline.apply(dataset, config, dataset_name)
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
            return None, None, None

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
            # if dataset_identifier != "disco-eth/EuroSpeech_subsetuk_splittrain":
            #     continue

            dataset = items["dataset"]
            config = items["config"]
            for _, example in enumerate(tqdm(dataset, desc=f"Creating: {out_dataset_name} from {dataset_identifier}", total=len(dataset))):
                audio_data = example[f"{config.audio_column}_features"]
                audio_path = self.save_audio_features(audio_data, audio_dir)

                target_value = example[config.target_column]
                if isinstance(target_value, str):
                    true_labels = [target_value.lower()]
                elif isinstance(target_value, list):
                    true_labels = [label.lower() for label in target_value]

                all_labels = example["all_labels"]
                all_labels = [label.lower() for label in all_labels]

                random.shuffle(all_labels)
                row = {
                    "source_dataset": dataset_identifier,
                    # "text": example["asr_transcript"],
                    "audio_features_path": audio_path,
                    "all_labels": all_labels,
                    "true_labels": true_labels,
                }
                dataset_list.append(row)

        random.shuffle(dataset_list)
        print("total_examples:", len(dataset_list))
        with open(os.path.join(save_path, output_dataset_file), "w") as f:
            json.dump(dataset_list, f, indent=4)
