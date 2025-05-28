from abc import ABC, abstractmethod
from itertools import product
from typing import Callable

import json
import random
from tqdm import tqdm
from vllm  import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel

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
                print(f"  → Dataset: {dataset_name}. Applying '{step_name}'...")
                dataset = func(dataset, config, **kwargs)
                print(f"    ✓ '{step_name}' completed")
            except Exception as e:
                print(f"    ✗ Dataset: {dataset_name}. Error in step '{step_name}': {e}")
                raise

        return dataset


class AudioProcessingPipeline(ProcessingPipeline):
    def _load_default_steps(self):
        self.add_step("normalize_audio", normalize_audio, batch_size=2)
        self.add_step("remove_special_characters", remove_special_characters)
        self.add_step("process_same_labels", process_same_labels)


SYSTEMMESSAGE = {
    "role": "user",
    "content": (
        "You are an advanced assistant trained to classify input text into relevant categories (labels) in English ONLY. \n"
        "Your task is to generate a JSON object with two fields:\n"
        "- 'true_labels': up to 10 labels that are accurate and contextually appropriate for the input.\n"
        "- 'false_labels': up to 10 incorrect but contextually challenging (hard negative) labels. These must be:\n"
        "   • Semantically or topically close to the true labels,\n, for example, spf-15 as true label and spf-30 as false"
        "   • Plausible but factually or contextually wrong,\n"
        "   • Never completely random, absurd, or trivially incorrect.\n\n if true label is president, false should be like vise-president, not python programming language (not random)"
        "There could be less then 10 labels if the text is short. The idea is that labels should be really related to the specific content"
        "The output must be a VALID JSON object, structured as:\n"
        '{"true_labels": ["..."], "false_labels": ["..."]}'
    ),
}

class OutputJSON(BaseModel):
    true_labels: list[str]
    false_labels: list[str]

class AnnotatorPipeline(ProcessingPipeline):
    def __init__(self, anotator: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.steps: list[tuple[str, Callable, dict]] = []
        self.init_anotator(anotator)
        self._load_default_steps()

    def _load_default_steps(self):
        self.add_step(
            "anotate_dataset",
            self.annotate_dataset
        )

    def init_anotator(self, anotator: str):
        self.tokenizer = AutoTokenizer.from_pretrained(anotator)
        self.llm = LLM(model=anotator, max_model_len = 8192, tensor_parallel_size=1, dtype="half", gpu_memory_utilization = 0.9, quantization = None)
        json_schema = OutputJSON.model_json_schema()
        guided_decoding_params_json = GuidedDecodingParams(json=json_schema)
        self.sampling_params = SamplingParams(temperature= 0.7 , repetition_penalty = 1.1, top_k=100, max_tokens=1024, top_p=0.8, stop="<end>",
                                              guided_decoding=guided_decoding_params_json)

    def get_text_to_label_few_shot_messages(self, text, examples):
        messages = [SYSTEMMESSAGE]

        example = random.choice(examples)
        input_message = {
            "role": "user",
            "content": (
                f'Here is an example input text: "{example["text"]}"\n'
                "Generate realistic true and hard false labels. English ONLY. \n"
                "Output in VALID JSON:\n"
                '{"true_labels": ["..."], "false_labels": ["..."]}'
            ),
        }
        messages.append(input_message)

        example_message = {
            "role": "assistant",
            "content": json.dumps({
                "true_labels": example["true_labels"],
                "false_labels": example["false_labels"]
            }, ensure_ascii=False)
        }
        messages.append(example_message)

        input_message = {
            "role": "user",
            "content": (
                f'Here is an input text: "{text}"\n'
                "Generate up to 10 true labels and up to 10 hard false labels. English ONLY. \n"
                "False labels must be related but factually or contextually incorrect (hard negatives).\n"
                "Output a VALID JSON:\n"
                '{"true_labels": ["..."], "false_labels": ["..."]}'
            ),
        }
        messages.append(input_message)

        return messages

    def annotate_batch(self, batch_chats: list[str]) -> list[dict]:
        outputs = self.llm.generate(batch_chats, sampling_params=self.sampling_params, use_tqdm=False)
        batch_results = [output.outputs[0].text for output in outputs]
        return batch_results

    def parse_batch_results(self, batch_results: list[str], batch_idxs: list[int]) -> list[dict]:
        parsed_results = []
        for id_, result in zip(batch_idxs, batch_results):
            try:
                parsed = json.loads(result)
                parsed_results.append((id_, parsed))
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON result [{id_}]: {e}")
            except Exception as e:
                print(f"Failed to parse result [{id_}]: {e}")
        return parsed_results

    def annotate_dataset(self, dataset: Dataset, config: DatasetConfig) -> str:
        dataset = dataset.select(range(6))

        batch_size = 2
        annotation_column = dataset[config.annotation_column]
        example_dataset = json.load(open('/home/werent4/DatasetGenerator/datasetgenerator/datasets/example/example_dataset.json', 'r', encoding='utf-8'))
        
        batch_idxs = []
        batch_chats = []

        parsed_results = []
        for idx, text in enumerate(tqdm(annotation_column, desc="Annotating dataset")):
            messages = self.get_text_to_label_few_shot_messages(text, example_dataset)
            chat =  self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_idxs.append(idx)
            batch_chats.append(chat)
            if len(batch_chats) == batch_size:
                try:
                    batch_results = self.annotate_batch(batch_chats)
                except Exception as err:
                    print(err)
                    continue

                parsed_results.extend(self.parse_batch_results(batch_results, batch_idxs))
                batch_idxs = []
                batch_chats = []

        if batch_chats:
            print("Called last batch with size:", len(batch_chats))
            try:
                batch_results = self.annotate_batch(batch_chats)
            except Exception as err:
                print(err)

            parsed_results.extend(self.parse_batch_results(batch_results, batch_idxs))
        
        dataset = self.update_dataset_with_annotations(dataset, parsed_results, config)
        return dataset

    def update_dataset_with_annotations(self, dataset: Dataset, parsed_results: list[tuple[int, dict]], config: DatasetConfig) -> Dataset:
        # Create dictionaries to hold the new values
        all_labels_dict = {idx: result["true_labels"] + result["false_labels"] 
                        for idx, result in parsed_results}
        
        true_labels_dict = {idx: result["true_labels"] 
                        for idx, result in parsed_results}
        
        def update_example(example, idx):
            if idx in all_labels_dict:
                example["all_labels"] = all_labels_dict[idx]
                example[config.target_column] = true_labels_dict[idx]  # rewrite with config
            else:
                example["all_labels"] = []
                example[config.target_column] = []
            return example
        
        return dataset.map(
            update_example, 
            with_indices=True,
            desc="Updating dataset with annotations"
        )

STR2PIPELINE = {
    "base": ProcessingPipeline,
    "audio": AudioProcessingPipeline,
    "annotator": AnnotatorPipeline
}

