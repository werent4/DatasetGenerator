from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DatasetConfig(ABC):
    dataset_name: str
    subset_name: str | list[str] | None = None
    split: str | None = None
    steps_to_skip: list[str] | None = None


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
