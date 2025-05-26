import numpy as np
from transformers import Wav2Vec2FeatureExtractor

from datasetgenerator.configs import AudioDatasetConfig
from datasets import Audio, Dataset, load_dataset


def normalize_audio(dataset: Dataset, config: AudioDatasetConfig, batch_size: int = 1) -> Dataset:
    max_duration = config.get_max_duration
    audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

    def normalize_batch(batch):
        processed_inputs = []

        for audio_data in batch[config.audio_column]:
            audio_array = audio_data["array"]
            if len(audio_array) > max_duration:
                audio_array = audio_array[:max_duration]
            elif len(audio_array) < max_duration:
                audio_array = np.pad(audio_array, (0, max_duration - len(audio_array)), mode="constant")
            inputs = audio_feature_extractor(audio_array, sampling_rate=config.sample_rate, return_tensors="pt")[
                "input_values"
            ].squeeze(0)

            processed_inputs.append(inputs.numpy())
        batch[f"{config.audio_column}_features"] = processed_inputs
        return batch

    return dataset.map(normalize_batch, batched=True, batch_size=batch_size)
