import hashlib
import json
import os
import random
from abc import ABC, abstractmethod
from itertools import product
import warnings

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

    @abstractmethod
    def extract_and_save_features(self, dataset, dataset_name, config, audio_dir):
        """Save the processed datasets."""
        pass

    def process(self, out_dataset_name: str = ""):
        """Process the datasets."""
        self.load_datasets()
        print("Datasets loaded.")

        save_path = os.path.join("/mnt/storage-werent4-1tb/generic-dataset", out_dataset_name)
        os.makedirs(save_path, exist_ok=True)

        for pipeline_name, pipeline  in self.pipelines:
            print(f"Applying pipeline: {pipeline.__class__.__name__}")
            pipeline.list_steps()

            for dataset_name, items in self.datasets.items():
                audio_dir = os.path.join(save_path, f"audio_features_{dataset_name.replace('/', '-')}")
                os.makedirs(audio_dir, exist_ok=True)

                config = items["config"]
                dataset = items["dataset"]
                
                if config.pipelines_to_skip is not None and pipeline_name in config.pipelines_to_skip:
                    print(f"Skipping pipeline {pipeline.__class__.__name__} for dataset {dataset_name} due to configuration.")  
                    continue

                dataset = self.extract_and_save_features(dataset, dataset_name, config, audio_dir)

                print(f"Processing dataset: {dataset_name}")
                processed_dataset = pipeline.apply(dataset, config, dataset_name)
                self.datasets[dataset_name]["dataset"] = processed_dataset
                print(f"Dataset {dataset_name} processed successfully.")

        self.save_datasets(save_path)

    @abstractmethod
    def save_datasets(self, save_path):
        """Save the processed datasets."""
        pass


class HuggingFaceProcessor(BaseProcessor):
    def _load_single_dataset(self, cfg, subset_name, split):
        """Load a single dataset configuration."""
        def add_index(example, idx):
            example["index"] = idx
            return example
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
            dataset = dataset.map(add_index, with_indices=True, batched=True, batch_size=16)

            # dataset = dataset.select(range(min(len(dataset), 50)))
            # warnings.warn("dataset cropped to 100 samples or len(dataset) for testing", UserWarning)

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

    def extract_and_save_features(self, dataset, dataset_name, config, audio_dir):   
        if dataset_name == "agkphysics/AudioSet_splittrain":
            dataset = dataset.select(range(15700))
            print("train cropped")

        if dataset_name == "agkphysics/AudioSet_splittest":
            dataset = dataset.select(range(6100))
            print("train cropped")

        audio_paths = []
        srs = []
        valid_indices = []
        skipped_count = 0
        
        print(f"Processing {len(dataset)} examples...")
        
        for i, example in enumerate(tqdm(dataset, desc=f"Saving {dataset_name} audio arrays")):
            try:
                idx = example["index"]
                audio_array = example[config.audio_column]['array']
                sampling_rate = example[config.audio_column]['sampling_rate']

                if audio_array is None:
                    warnings.warn(f"None audio array found for index {idx}. Skipping.", UserWarning)
                    skipped_count += 1
                    continue
                    
                srs.append(sampling_rate)
                if len(audio_array) == 0:
                    warnings.warn(f"Empty audio array found for index {idx}. Creating a zero array.", UserWarning)
                    audio_array = np.zeros(16000) 
                    
                if isinstance(audio_array, np.ndarray):
                    audio_array = torch.from_numpy(audio_array)
                elif isinstance(audio_array, list):
                    audio_array = torch.tensor(audio_array)
                elif isinstance(audio_array, torch.Tensor):
                    pass  
                else:
                    print(f"Warning: Unknown audio_array type {type(audio_array)} for index {idx}")
                    audio_array = torch.tensor(audio_array)
                    
                if isinstance(audio_array, torch.Tensor):
                    audio_hash = hashlib.md5(audio_array.numpy().tobytes()).hexdigest()
                else:
                    audio_hash = hashlib.md5(audio_array.tobytes()).hexdigest()
                    
                audio_filename = f"{audio_hash}-{sampling_rate}.pt"
                audio_path = os.path.join(audio_dir, audio_filename)
                
                torch.save(audio_array, audio_path)
                audio_paths.append(audio_path)
                valid_indices.append(i)
                
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                print(f"Skipping corrupted audio file at index {i}")
                skipped_count += 1
                continue
        
        print(f"Processed: {len(valid_indices)} valid examples")
        print(f"Skipped: {skipped_count} corrupted/invalid examples")
        
        dataset = dataset.select(valid_indices)
        
        if "audio_path" in dataset.column_names:
            dataset = dataset.remove_columns(["audio_path"])
        if "sampling_rate" in dataset.column_names:
            dataset = dataset.remove_columns(["sampling_rate"])
            
        dataset = dataset.add_column("audio_path", audio_paths)
        dataset = dataset.add_column("sampling_rate", srs)
        
        print(f"Dataset size after aduio processing: {len(dataset)} examples")
        return dataset
    
    def save_datasets(self, save_path):
        """Save the processed datasets to disk."""

        for dataset_identifier, items in self.datasets.items():

            dataset_list = []
            dataset = items["dataset"]
            config = items["config"]
            for example in tqdm(dataset, desc=f"Creating: GliCLass dataset from {dataset_identifier}", total=len(dataset)):
                idx = example["index"]   
                audio_path = example["audio_path"]
                sample_rate = example["sampling_rate"]

                target_value = example[config.target_column]
                if isinstance(target_value, str):
                    true_labels = [target_value.lower()]
                elif isinstance(target_value, list):
                    true_labels = [label.lower() for label in target_value]

                all_labels = example["all_labels"]
                all_labels = [label.lower() for label in all_labels]

                random.shuffle(all_labels)
                row = {
                    "id": idx,
                    "source_dataset": dataset_identifier,
                    "audio_path": audio_path,
                    "sample_rate" : sample_rate,
                    "all_labels": all_labels,
                    "true_labels": true_labels,
                }
                if config.annotation_column:
                    row["text"] = example[config.annotation_column]
                dataset_list.append(row)

            random.shuffle(dataset_list)
            print("total_examples:", len(dataset_list))
            with open(os.path.join(save_path, f"{dataset_identifier.replace('/', '-')}.json"), "w") as f:
                json.dump(dataset_list, f, indent=4)

            print("data saved to: ", os.path.join(save_path, f"{dataset_identifier.replace('/', '-')}.json"))
