import os
if os.path.exists("/mnt/werent4-storage"):
    os.environ['HF_HOME'] = '/mnt/werent4-storage/huggingface_cache'
    os.environ['TRANSFORMERS_CACHE'] = '/mnt/werent4-storage/huggingface_cache'
    os.environ['HF_DATASETS_CACHE'] = '/mnt/werent4-storage/huggingface_cache'

import hashlib
import json

import random
from abc import ABC, abstractmethod
from itertools import product

import numpy as nps
import torch
from tqdm import tqdm

from datasetgenerator.configs import AudioDatasetConfig, DatasetConfig, load_configs
from datasetgenerator.pipelines import (
    AudioProcessingPipeline
)
from datasetgenerator.processors import HuggingFaceProcessor
from datasets import Audio, load_dataset

# configs = load_configs("datasetgenerator/configs/configs/gliclass-audio.json", "audio")
pipelines, configs = load_configs("configs/configs/dataset/test.json", "configs/configs/processor/huggingface_processor.json", "audio")
processor= HuggingFaceProcessor(configs, pipelines=pipelines)
processor.process(out_dataset_name="annotations-cv-mozila")
