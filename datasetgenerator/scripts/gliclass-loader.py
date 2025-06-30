import hashlib
import io
import json
import math
import os
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import backoff
import uuid
import librosa
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfFileSystem, get_token, hf_hub_url, list_repo_files
from tqdm import tqdm

# v1
def create_label_dataset(df: pd.DataFrame, 
                        min_additional_labels: int = 2, 
                        max_additional_labels: int = 8) -> pd.DataFrame:
    result_df = df.copy()
    result_df = result_df.rename(columns={'caption': 'text'})
    all_captions = df['caption'].unique().tolist()
    true_labels_list = []
    all_labels_list = []
    
    for idx, row in result_df.iterrows():
        current_caption = row['text']
        
        true_labels = [current_caption]
        
        num_additional = random.randint(min_additional_labels, max_additional_labels)
        other_captions = [cap for cap in all_captions if cap != current_caption]
        
        if len(other_captions) >= num_additional:
            additional_labels = random.sample(other_captions, num_additional)
        else:
            additional_labels = other_captions
        
        all_labels = true_labels + additional_labels
        
        random.shuffle(all_labels)
        
        true_labels_list.append(true_labels)
        all_labels_list.append(all_labels)
    
    result_df['true_labels'] = true_labels_list
    result_df['all_labels'] = all_labels_list
    
    return result_df

# v2
def create_label_dataset_sets(df: pd.DataFrame,
                                   min_additional_labels: int = 2,
                                   max_additional_labels: int = 8) -> pd.DataFrame:
    result_df = df.copy()
    result_df = result_df.rename(columns={'caption': 'text'})
    
    all_captions_set = set(df['caption'].unique())
    print(f"Found {len(all_captions_set)} unique captions")
    
    def generate_labels_fast(text):
        true_labels = [text]
        num_additional = random.randint(min_additional_labels, max_additional_labels)
        
        other_captions_set = all_captions_set - {text}
        
        if len(other_captions_set) >= num_additional:
            other_captions_list = list(other_captions_set)
            additional_labels = random.sample(other_captions_list, num_additional)
        else:
            additional_labels = list(other_captions_set)
            
        all_labels = true_labels + additional_labels
        random.shuffle(all_labels)
        
        return true_labels, all_labels
    
    tqdm.pandas(desc="Generating labels")
    label_data = result_df['text'].progress_apply(generate_labels_fast)
    
    result_df['true_labels'] = [x[0] for x in label_data]
    result_df['all_labels'] = [x[1] for x in label_data]
    
    return result_df

# v3
def create_label_dataset_sets_scalable(df, min_additional_labels=2, max_additional_labels=8):
    print("v3")
    df = df.rename(columns={'caption': 'text'})
    
    unique_captions = df['text'].unique()
    caption_to_idx = {cap: idx for idx, cap in enumerate(unique_captions)}
    n_unique = len(unique_captions)
    
    print(f"Found {n_unique} unique captions out of {len(df)} rows")
    
    def generate_labels_for_text(text):
        true_labels = [text]
        current_idx = caption_to_idx[text]
        
        available_indices = np.concatenate([
            np.arange(current_idx),
            np.arange(current_idx + 1, n_unique)
        ])
        
        num_additional = random.randint(min_additional_labels, max_additional_labels)
        num_to_select = min(num_additional, len(available_indices))
        
        selected_indices = np.random.choice(available_indices, size=num_to_select, replace=False)
        additional_labels = [unique_captions[idx] for idx in selected_indices]
        
        all_labels = true_labels + additional_labels
        random.shuffle(all_labels)
        
        return true_labels, all_labels
    
    tqdm.pandas(desc="Generating labels")
    label_data = df['text'].progress_apply(generate_labels_for_text)
    
    df['true_labels'] = [x[0] for x in label_data]
    df['all_labels'] = [x[1] for x in label_data]
    
    return df

#v4
def create_label_dataset_sets_scalable_4(df, min_additional_labels=2, max_additional_labels=8):
    print("v4")
    df = df.rename(columns={'caption': 'text'})
    
    unique_captions = df['text'].unique()
    caption_to_idx = {cap: idx for idx, cap in enumerate(unique_captions)}
    n_unique = len(unique_captions)
    
    print(f"Found {n_unique} unique captions out of {len(df)} rows")
    
    def get_idx(current_idx, selected):
        while True:
            idx = random.randint(0, n_unique-1)
            if idx != current_idx and idx not in selected:
                return idx

    def generate_labels_for_text(text):
        true_labels = [text]
        current_idx = caption_to_idx[text]

        num_additional = random.randint(min_additional_labels, max_additional_labels)
        selected_indices = []
        
        for _ in range(num_additional):
            selected_indices.append(get_idx(current_idx, selected_indices))

        additional_labels = [unique_captions[idx] for idx in selected_indices]
        
        all_labels = true_labels + additional_labels
        random.shuffle(all_labels)
        
        return true_labels, all_labels
    
    tqdm.pandas(desc="Generating labels")
    label_data = df['text'].progress_apply(generate_labels_for_text)
    
    df['true_labels'] = [x[0] for x in label_data]
    df['all_labels'] = [x[1] for x in label_data]
    
    return df

def extract_and_save_features(dataset, audio_dir):   
    audio_paths = []
    srs = []
    valid_indices = []
    skipped_count = 0
    
    print(f"Extracting {len(dataset)} audio arrays...")
    for i, (idx, row) in enumerate(tqdm(dataset.iterrows(), total=len(dataset))):
        try:
            idx = row["id"]
            audio_array = row["audio"]
            sample_rate = row["sample_rate"]
            srs.append(sample_rate)
            is_zeros = False
            if len(audio_array) == 0:
                is_zeros = True
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

        #   if isinstance(audio_array, torch.Tensor):
        #       audio_hash = hashlib.md5(audio_array.numpy().tobytes()).hexdigest()
        #   else:
        #       audio_hash = hashlib.md5(audio_array.tobytes()).hexdigest()
            base_filename = f"{uuid.uuid4().hex}-{sample_rate}"
            if is_zeros:
                audio_filename = f"{base_filename}-zeros.pt"
            else:
                audio_filename = f"{base_filename}.pt"
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
    
    filtered_dataset = dataset.iloc[valid_indices].copy()
    
    columns_to_remove = ['audio_path', 'sampling_rate', 'audio']
    for col in columns_to_remove:
        if col in filtered_dataset.columns:
            filtered_dataset = filtered_dataset.drop(columns=[col])
    
    filtered_dataset['audio_path'] = audio_paths
    filtered_dataset['sampling_rate'] = srs
    
    print(f"Dataset size after audio processing: {len(filtered_dataset)} examples")
    return filtered_dataset

def save_dataset(dataset, save_path, dataset_identifier = "laion/LAION-Audio-300M_splittrain"):
  dataset_list = []
  for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
      idx = row["id"]   
      audio_path = row["audio_path"]
      text = row["text"]
      sample_rate = row["sampling_rate"]
      true_labels = row["true_labels"]
      all_labels = row["all_labels"]

      if isinstance(true_labels, str):
          true_labels = [true_labels.lower()]
      elif isinstance(true_labels, list):
          true_labels = [label.lower() for label in true_labels]
      all_labels = [label.lower() for label in all_labels]

      random.shuffle(all_labels)
      row = {
          "id": idx,
          "source_dataset": dataset_identifier,
          "audio_path": audio_path,
          "text": text,
          "sample_rate" : sample_rate,
          "all_labels": all_labels,
          "true_labels": true_labels,
      }
      dataset_list.append(row)

  random.shuffle(dataset_list)
  print("total_examples:", len(dataset_list))
  with open(os.path.join(save_path, f"{dataset_identifier.replace('/', '-')}.json"), "w") as f:
      json.dump(dataset_list, f, indent=4)

  print("data saved to: ", os.path.join(save_path, f"{dataset_identifier.replace('/', '-')}.json"))

class DatasetLoader:
  def __init__(self, repo_id: str):
    self.repo_id = repo_id
    self.tars = self.load_tars()
    print("Total tars: ", len(self.tars))

  def load_tars(self):
    files = list_repo_files(self.repo_id, repo_type="dataset")
    tar_files = [f for f in files if f.endswith('.tar')]
    return sorted(tar_files)

  @backoff.on_exception(
      backoff.expo,
      Exception,
      max_tries=5,
      max_time=120,
      jitter=backoff.full_jitter
  )
  def sample_from_tar(self, tar_file: str, samples_per_tar: int = 100):
    time.sleep(random.uniform(0.3, 1.5))
    dataset = load_dataset(
      self.repo_id, 
      data_files=tar_file,
      streaming=True
    )
      
    samples = []
    train_data = dataset["train"]
    limited_data = train_data.take(samples_per_tar)

    for i, sample in enumerate(limited_data):
      if i >= samples_per_tar:
        break

      audio_data = sample['audio.mp3']["array"]
      sample_rate = sample['audio.mp3']["sampling_rate"]
      caption = sample['metadata.json']["caption"]
      
      samples.append({
        "id": f"{tar_file}_{i}",
        'audio': audio_data,
        "sample_rate" : sample_rate,
        'caption': caption,
      })
    
    return pd.DataFrame(samples)


  def _process_tar_chunk(self, tar_chunk: List[str], samples_per_tar: int, thread_id: int) -> pd.DataFrame:
    thread_samples = []
        
    with tqdm(tar_chunk, desc=f"Thread-{thread_id}", position=thread_id, leave=False) as pbar:
      for tar_file in pbar:
        pbar.set_description(f"Thread-{thread_id}: {tar_file.split('/')[-1]}")
                
        try:
          df = self.sample_from_tar(tar_file, samples_per_tar)
          if not df.empty:
            thread_samples.append(df)
            pbar.set_postfix(loaded=sum(len(df) for df in thread_samples))
        except Exception as e:
          print(f"Final failure for {tar_file}: {e}")
          continue
                
    return pd.concat(thread_samples, ignore_index=True) if thread_samples else pd.DataFrame()


  def load_subset(self, total_size= 1000, split_accros_tars = True):
    if not split_accros_tars:
        return self.sample_from_tar(self.tars[0], total_size)
    
    data_per_tar = math.ceil(total_size / len(self.tars))
    print(f"Loading ~{data_per_tar} samples per tar file to reach total of {total_size}")

    all_samples = []
    total_loaded = 0

    for i, tar in enumerate(tqdm(self.tars, desc="sampling data")):
      if total_loaded >= total_size:
          print(f"Reached target size: {total_loaded} samples")
          break
      
      remaining = total_size - total_loaded
      samples_to_load = min(data_per_tar, remaining)

      print(f"Processing tar {i+1}/{len(self.tars)}: {tar}")
      print(f"Loading {samples_to_load} samples (total loaded: {total_loaded}/{total_size})")
      
      df = self.sample_from_tar(tar, samples_to_load)

      if not df.empty:
        all_samples.append(df)
        total_loaded += len(df)
        print(f"Successfully loaded {len(df)} samples from {tar}")
      else:
        print(f"No data loaded from {tar}")
  
      if total_loaded >= total_size:
        print(f"Reached target size: {total_loaded} samples")
        break

    if all_samples:
      result_df = pd.concat(all_samples, ignore_index=True)       
      print(f"Final dataset size: {len(result_df)} samples")
      return result_df
    else:
      print("No data was loaded")
      return pd.DataFrame()

class WebDatasetLoader:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.fs = HfFileSystem()
        self.setup_urls()
    
    def setup_urls(self):
        files = [self.fs.resolve_path(path) for path in self.fs.glob(f"hf://datasets/{self.repo_id}/**/*.tar")]
        self.urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
        print(f"Found {len(self.urls)} tar files")
        random.shuffle(self.urls) 
        # self.urls = self.urls[900:1100]

    def create_webdataset_url(self, url_subset):
        token = get_token()
        urls_string = '::'.join(url_subset)
        return f"pipe: curl -s -L -H 'Authorization:Bearer {token}' {urls_string}"
    
    def decode_audio(self, data):
        try:
            audio_data, sample_rate = librosa.load(io.BytesIO(data), sr=None)
            return {"array": audio_data, "sampling_rate": sample_rate}
        except Exception as e:
            print(f"Error decoding audio: {e}")
            return {"array": np.array([]), "sampling_rate": 16000}
        
    def decode_metadata(self, data):
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            return json.loads(data)
        if isinstance(data, bytearray | bytes):
            try:
                return json.loads(data.decode("utf-8"))
            except Exception as e:
                print(f"Error decoding metadata: {e}")
                return {"caption": ""}

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=5,
        max_time=120,
        jitter=backoff.full_jitter
    )
    def sample_from_tar_urls(self, url, samples_per_tar):
        time.sleep(random.uniform(0.1, 0.5))
        
        try:
            urls_with_auth = self.create_webdataset_url(url)
            
            dataset = (wds.WebDataset(urls_with_auth, shardshuffle=False)
                      .decode()
                      .to_tuple("audio.mp3", "metadata.json")
                      .map(lambda x: (self.decode_audio(x[0]), self.decode_metadata(x[1]))))
            
            samples = []
            for i, (audio_data, metadata) in enumerate(dataset):
                if i >= samples_per_tar:
                    break
                    
                sample_id = f"{url[0].split('/')[-1]}_{i}"
                samples.append({
                    "id": sample_id,
                    'audio': audio_data["array"],
                    "sample_rate": audio_data["sampling_rate"],
                    'caption': metadata.get("caption", "<<NOTFOUND>>"),
                })
            
            return pd.DataFrame(samples)
            
        except Exception as e:
            print(f"Error processing tar batch: {e}")
            return pd.DataFrame()

    def load_subset(self, total_size, audio_dir, split_accros_tars = True):
        if not split_accros_tars:
            raise ValueError("I didnt implemented specific splitting yet")
        
        data_per_tar = math.ceil(total_size / len(self.urls))
        print(f"Loading ~{data_per_tar} samples per tar file to reach total of {total_size}")

        all_samples = []
        total_loaded = 0

        for i, url in enumerate(tqdm(self.urls, desc="sampling data")):
            if total_loaded >= total_size:
                print(f"Reached target size: {total_loaded} samples")
                break

            remaining = total_size - total_loaded
            samples_to_load = min(data_per_tar, remaining)

            df = self.sample_from_tar_urls([url], samples_to_load)
            if not df.empty:
                audio_paths_saved_subset = extract_and_save_features(df, audio_dir)
                all_samples.append(audio_paths_saved_subset)
                total_loaded += len(audio_paths_saved_subset)
                print(f"Successfully loaded {len(audio_paths_saved_subset)} samples from batch")
            else:
                print(f"No data loaded from batch")

            if total_loaded >= total_size:
                print(f"Reached target size: {total_loaded} samples")
                break
            
        if all_samples:
            result_df = pd.concat(all_samples, ignore_index=True)
            print(f"Final dataset size: {len(result_df)} samples")
            return result_df
        else:
            print("No data was loaded")
            return pd.DataFrame()


if __name__ == "__main__":
  dataset_identifier = "laion/LAION-Audio-300M_splittrain"
  dataset_name = "laion/LAION-Audio-300M"
  samples_count = 1_001_000

  save_path = "/mnt/storage-werent4-2tb/generic-dataset/LAION-Audio-300M_splittrain-1M"
  audio_dir = os.path.join(save_path, f"audio_features_{dataset_identifier.replace('/', '-')}")

  os.makedirs(save_path, exist_ok= True)
  os.makedirs(audio_dir, exist_ok= True)

  dataset_loader = WebDatasetLoader(dataset_name)
  sampled_subset = dataset_loader.load_subset(samples_count, audio_dir, split_accros_tars= True)
  labels_sampled_subset = create_label_dataset_sets_scalable_4(sampled_subset, max_additional_labels= 20)

  save_dataset(labels_sampled_subset, save_path, dataset_identifier)
