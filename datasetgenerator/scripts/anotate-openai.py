import os
import openai
import json
import random
from tqdm import tqdm
from typing import List
from openai import AzureOpenAI
from dotenv import load_dotenv
import time
import threading
import tiktoken

load_dotenv()
endpoint = "https://knowledgator-gpt4.openai.azure.com/"
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

NUM_TRUE = 25
NUM_FALSE = 25
SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are an advanced assistant trained to classify input text into relevant categories (labels). "
        "Your task is to generate a JSON object with two fields:\n"
        f"- 'true_labels': {NUM_TRUE} labels that are accurate and contextually appropriate for the input.\n"
        f"- 'false_labels': {NUM_FALSE} incorrect but contextually challenging (hard negative) labels.\n"
        "False labels must be semantically or topically close to the true ones, but incorrect. Do not just negate true labels, for example, instead of 'Not fast-acting' as a false label use 'slow-acting'\n"
        "The output must be a valid JSON object: {'true_labels': [...], 'false_labels': [...]}"
    )
}

def get_processed_ids_from_file(filename):
    processed_ids = set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    processed_ids.add(record['id'])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print("Output file doesn't exist - starting fresh")
        return set()
    
    return processed_ids

def get_text_to_label_few_shot_messages(text, examples, max_shots=5):
    messages = [SYSTEM_MSG]

    buckets = [0, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    text_length = len(text.split())
    for i, bucket in enumerate(buckets):
        if text_length <= bucket:
            bucket_id = f"{buckets[i-1]}-{buckets[i]}"
            break
    if text_length < 48:
        max_shots = 4
    elif text_length < 192:
        max_shots = 2
    else:
        max_shots = 1

    random.shuffle(examples[bucket_id])
    few_shots = examples[bucket_id][:max_shots]

    for example in few_shots:
        messages.append({
            "role": "user",
            "content": (
                f'Here is an example input text: "{example["text"]}"\n'
                "Generate realistic true and hard false labels.\n"
                "Output in VALID JSON:\n"
                '{"true_labels": ["..."], "false_labels": ["..."]}'
            )
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "true_labels": example["true_labels"],
                "false_labels": example["false_labels"]
            }, ensure_ascii=False)
        })

    messages.append({
        "role": "user",
        "content": (
            f'Here is an input text: "{text}"\n'
            f"Generate {NUM_TRUE} true labels and {NUM_FALSE} hard false labels.\n"
            "False labels must be related but factually or contextually incorrect (hard negatives).\n"
            "Output a VALID JSON:\n"
            '{"true_labels": ["..."], "false_labels": ["..."]}'
        )
    })
    return messages

class OpenAIAnnotator:
    def __init__(self, client, example_dataset):
        self.tokens_per_m = 200_000
        self.rpm = 2000
        self.max_out_tokens = 1024
        self.client = client
        self.example_dataset = example_dataset
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        self.file_lock = threading.Lock()

    def estimate_tokens(self, text):
        messages = get_text_to_label_few_shot_messages(text, self.example_dataset, max_shots=5)
        
        input_tokens = 0
        for msg in messages:
            input_tokens += len(self.encoding.encode(msg["content"])) + 4 # +4 for metadata tokens 
        
        return input_tokens + self.max_out_tokens

    def request_openai(self, messages, model, temperature=0.5):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_out_tokens,
            top_p=0.8,
            n=1
        )
        return response.choices[0].message.content

    def get_save_batches(self, data, processed_indexes):
        if processed_indexes is None:
            processed_indexes = set()
        else:
            processed_indexes = set(processed_indexes)

        batches = []
        current_batch = []
        current_batch_tokens = 0
        current_batch_requests = 0

        max_requests_per_batch = self.rpm
        max_tokens_per_batch = self.tokens_per_m

        for row in data:
            if row["id"] in processed_indexes:
                continue
            estimated_tokens = self.estimate_tokens(row["text"])      
            
            if (current_batch_requests + 1 > max_requests_per_batch or 
                current_batch_tokens + estimated_tokens > max_tokens_per_batch):

                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0
                    current_batch_requests = 0
            
            current_batch.append(row)
            current_batch_tokens += estimated_tokens
            current_batch_requests += 1

        if current_batch:
            batches.append(current_batch)

        return batches

    def process_single_item(self, item, delay, output_file):       
        time.sleep(delay)
        
        idx = item["id"]
        text = item["text"]
        
        try:
            messages = get_text_to_label_few_shot_messages(text, self.example_dataset, max_shots=5)
            result = self.request_openai(messages, model=deployment)
            parsed = json.loads(result)
            final_row = {**item, **parsed}
            
            self.save_result(final_row, output_file)
            
            if self.progress_bar:
                self.progress_bar.update(1)
                self.progress_bar.set_postfix({"Status": f"Item {idx}"})
            
        except Exception as e:
            print("Failed to proccess index: ", idx)
            if self.progress_bar:
                self.progress_bar.update(1)
                self.progress_bar.set_postfix({"Status": f"Item {idx} Failed"})

    def save_result(self, result, filename):
        """Thread-safe запись результата"""
        with self.file_lock:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

    def generate_dataset(self, data, output_file, processed_indexes= None):
        save_batches = self.get_save_batches(data, processed_indexes)
        print("Created: ", len(save_batches), "to be proccessed")

        if processed_indexes:
            total = len(data) - len(processed_indexes)
        else:
            total = len(data)

        with tqdm(total=total, desc="Processing", unit="items") as pbar:
            self.progress_bar = pbar
            for batch_idx, batch in enumerate(save_batches):
                print(f"Processing batch {batch_idx + 1}/{len(save_batches)} ({len(batch)} items)")
                
                threads = []
                for item_idx, item in enumerate(batch):
                    delay = random.uniform(0, 60)
                    thread = threading.Thread(
                        target=self.process_single_item,
                        args=(item, delay, output_file)
                    )
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()
        
        self.progress_bar = None
        print(f"Dataset generation completed!\nData saved to: {output_file}")

    

if __name__ == "__main__":
    out_name = "AbstractTTS-PODCAST-gliclass.jsonl"
    example_dataset = json.load(open('examples.json', 'r', encoding='utf-8'))

    data_path = "/mnt/werent4-storage/AbstractTTS-PODCAST/PODCAST-pre-saved.json"
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    data = [row for row in data 
            if row.get("text") and isinstance(row["text"], str) and row["text"].strip()]
    data = data[:75000]

    processed_ids = get_processed_ids_from_file(out_name)

    print("Already have: ", len(processed_ids), "processed ids")
    annotator = OpenAIAnnotator(
        client, example_dataset
    )
    annotator.generate_dataset(data, out_name, processed_indexes= processed_ids)