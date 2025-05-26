import json
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DatasetConfig(ABC):
    dataset_name: str
    subset_name: str | list[str] | None = None
    split: str | None = None


@dataclass
class AudioDatasetConfig(DatasetConfig):
    audio_column: str = "audio"
    target_column: str = "label"
    sample_rate: int = 16000
    max_duration_s: int = 5

    @property
    def get_max_duration(self) -> int:
        return self.max_duration_s * self.max_duration_s


TYPE2CONFIG = {"base": DatasetConfig, "audio": AudioDatasetConfig}


def load_configs(file_path: str, config_type: str) -> list[DatasetConfig]:
    if config_type not in TYPE2CONFIG:
        raise ValueError(f"Unsupported config type: {config_type}")

    with open(file_path) as file:
        configs = json.load(file)

    return [TYPE2CONFIG[config_type](**config) for config in configs]
