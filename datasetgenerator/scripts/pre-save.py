import os
if os.path.exists("/mnt/werent4-storage"):
    os.environ['HF_HOME'] = '/mnt/werent4-storage/huggingface_cache'
    os.environ['TRANSFORMERS_CACHE'] = '/mnt/werent4-storage/huggingface_cache'
    os.environ['HF_DATASETS_CACHE'] = '/mnt/werent4-storage/huggingface_cache'

import warnings
import json
import torch
import uuid
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

def add_id(example, idx):
    example['id'] = idx
    return example

def extract_and_save_features(dataset, audio_dir):   
    audio_paths = []
    srs = []
    valid_indices = []
    skipped_count = 0
    
    print(f"Extracting {len(dataset)} audio arrays...")
    for i, (row) in enumerate(tqdm(dataset, total=len(dataset))):
        try:
            idx = row["id"]
            audio_array = row["audio"]["array"]
            sample_rate = row["audio"]["sampling_rate"]
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
    
    filtered_dataset = dataset.select(valid_indices)
    
    columns_to_remove = ['audio_path', 'sampling_rate', 'audio']
    existing_columns_to_remove = [col for col in columns_to_remove if col in filtered_dataset.column_names]
    if existing_columns_to_remove:
        filtered_dataset = filtered_dataset.remove_columns(existing_columns_to_remove)
    
    filtered_dataset = filtered_dataset.add_column('audio_path', audio_paths)
    filtered_dataset = filtered_dataset.add_column('sampling_rate', srs)
    
    print(f"Dataset size after audio processing: {len(filtered_dataset)} examples")
    return filtered_dataset

def pre_save(ds_with_path, save_dir, text_col):
    lines = []
    for row in tqdm(ds_with_path, desc= "creating pre-saved verison of dataset"):
        idx = row["id"]
        audio_path = row['audio_path']
        sampling_rate = row['sampling_rate']
        text = row[text_col]   
        lines.append({
          "id": idx,
          "source_dataset": "AbstractTTS/PODCAST_splittrain",
          "audio_path": audio_path,
          "text": text,
          "sample_rate" : sampling_rate,
        })
         
    with open(os.path.join(save_dir, "PODCAST-pre-saved"), "w", encoding= "utf-8") as f:
        json.dump(lines, f, indent= 4)
    print("saved to:", os.path.join(save_dir, "PODCAST-pre-saved"))

if __name__ == "__main__":
    dataset_name = "AbstractTTS/PODCAST"
    
    save_dir = "/mnt/werent4-storage/AbstractTTS-PODCAST"
    audio_dir = "/mnt/werent4-storage/AbstractTTS-PODCAST/audio_features_AbstractTTS-PODCAST"
    os.makedirs(save_dir, exist_ok= True)
    os.makedirs(audio_dir, exist_ok= True)
    
    
    dataset = load_dataset(dataset_name, split= 'train')
    dataset = dataset.map(add_id, with_indices=True, batch_size= 1000)
    ds_with_path = extract_and_save_features(dataset, audio_dir)
    pre_save(ds_with_path, save_dir, "transcription")