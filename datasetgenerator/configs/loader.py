import json
from datasetgenerator.pipelines import ProcessingPipeline, STR2PIPELINE

from .dataset_configs import TYPE2CONFIG, DatasetConfig


def load_configs(
    datasets_config: str, processor_config: str, config_type: str
) -> tuple[list[ProcessingPipeline], list[DatasetConfig]]:
    if config_type not in TYPE2CONFIG:
        raise ValueError(f"Unsupported config type: {config_type}")

    with open(processor_config) as file:
        processor_config = json.load(file)
    pipelines = []
    for pipeline in processor_config["pipelines"]:
        if pipeline not in STR2PIPELINE:
            raise ValueError(f"Unsupported pipeline: {pipeline}")
        pipelines.append((pipeline, STR2PIPELINE[pipeline]()))

    with open(datasets_config) as file:
        configs = json.load(file)

    return pipelines, [TYPE2CONFIG[config_type](**config) for config in configs]
