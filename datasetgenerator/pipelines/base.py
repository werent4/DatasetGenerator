from abc import ABC, abstractmethod
from itertools import product
from typing import Callable

from datasets import Dataset

from datasetgenerator.configs import DatasetConfig

from .audio_transforms import normalize_audio
from .text_transforms import process_same_labels, remove_special_characters


class ProcessingPipeline(ABC):
    def __init__(self):
        self.steps: list[tuple[str, Callable, dict]] = []
        self._load_default_steps()

    @abstractmethod
    def _load_default_steps(self):
        """Load default processing steps for the pipeline."""
        raise NotImplementedError("Subclasses must implement _load_default_steps")

    def add_step(self, name: str, func: Callable, **kwargs):
        """Add a processing step to the pipeline."""
        self.steps.append((name, func, kwargs))
        print(f"Added processing step: {name}")

    def remove_step(self, name: str):
        """Remove a processing step from the pipeline."""
        original_length = len(self.steps)
        self.steps = [step for step in self.steps if step[0] != name]
        if len(self.steps) < original_length:
            print(f"Removed processing step: {name}")
        else:
            print(f"Step '{name}' not found")

    def clear(self):
        """Clear all processing steps."""
        self.steps.clear()
        print("Cleared all processing steps")

    def list_steps(self):
        """List all processing steps."""
        if not self.steps:
            print("No processing steps defined")
            return

        print("Processing pipeline steps:")
        for i, (name, _, kwargs) in enumerate(self.steps, 1):
            kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else "no args"
            print(f"  {i}. {name} ({kwargs_str})")

    def apply(self, dataset: Dataset, config: DatasetConfig, dataset_name: str = "") -> Dataset:
        """Apply all pipeline steps to a dataset."""
        if not self.steps:
            print(f"No processing steps for {dataset_name}")
            return dataset

        print(f"Applying {len(self.steps)} processing steps to {dataset_name}")

        for step_name, func, kwargs in self.steps:
            if config.steps_to_skip is not None and step_name in config.steps_to_skip:
                print(f"  → Skipping step '{step_name}' as per configuration")
                continue

            try:
                print(f"  → Applying '{step_name}'...")
                dataset = func(dataset, config, **kwargs)
                print(f"    ✓ '{step_name}' completed")
            except Exception as e:
                print(f"    ✗ Error in step '{step_name}': {e}")
                raise

        return dataset


class AudioProcessingPipeline(ProcessingPipeline):
    def _load_default_steps(self):
        self.add_step("normalize_audio", normalize_audio, batch_size=2)
        self.add_step("remove_special_characters", remove_special_characters)
        self.add_step("process_same_labels", process_same_labels)
