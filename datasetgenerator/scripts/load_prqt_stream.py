from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd
import warnings
import numpy as np
import torch
import uuid
import os
import random
import json

def process_labels(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, list):
        return x
    else:
        print("lol 1 label")
        return [x] 

def create_label_dataset_sets_scalable_4(df, text_col, min_additional_labels=2, max_additional_labels=8):
    print("v4")
    tqdm.pandas(desc="Collecting unique labels")
    all_labels = set()
    for label_list in df[text_col].progress_apply(process_labels):
        all_labels.update(label_list)

    unique_labels = list(all_labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    n_unique = len(label_to_idx)
    print(f"Found {n_unique} unique labels out of {len(df)} rows")
    
    def get_idx(current_idx, selected):
        while True:
            idx = random.randint(0, n_unique-1)
            if idx not in current_idx and idx not in selected:
                return idx

    def generate_labels_for_text(text_list):
        if isinstance(text_list, np.ndarray):
            true_labels = text_list.tolist()
        elif isinstance(text_list, list):
            true_labels = text_list.copy()
        else:
            true_labels = [text_list]

        current_idxs = [label_to_idx[label] for label in true_labels]

        num_additional = random.randint(min_additional_labels, max_additional_labels)
        selected_indices = []
        
        for _ in range(num_additional):
            selected_indices.append(get_idx(current_idxs, selected_indices))

        additional_labels = [unique_labels[idx] for idx in selected_indices]
        
        all_labels = true_labels + additional_labels
        random.seed(42)
        random.shuffle(all_labels)
        
        return true_labels, all_labels
    
    tqdm.pandas(desc="Generating labels")
    label_data = df[text_col].progress_apply(generate_labels_for_text)
    
    df['true_labels'] = [x[0] for x in label_data]
    df['all_labels'] = [x[1] for x in label_data]
    
    return df

def save_dataset(dataset, save_path, dataset_identifier):
    dataset_list = []
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
        idx = row["id"]   
        audio_path = row["audio_path"]
        sample_rate = row["sampling_rate"]
        true_labels = row["true_labels"]
        all_labels = row["all_labels"]

        if isinstance(true_labels, str):
            true_labels = [true_labels.lower()]
        elif isinstance(true_labels, list):
            true_labels = [label.lower() for label in true_labels]
        all_labels = [label.lower() for label in all_labels]

        random.seed(42)
        random.shuffle(all_labels)
        row = {
            "id": idx,
            "source_dataset": dataset_identifier,
            "audio_path": audio_path,
            "sample_rate" : sample_rate,
            "all_labels": all_labels,
            "true_labels": true_labels,
        }
        dataset_list.append(row)

    random.seed(42)
    random.shuffle(dataset_list)
    print("total_examples:", len(dataset_list))
    with open(os.path.join(save_path, f"{dataset_identifier.replace('/', '-')}.json"), "w") as f:
        json.dump(dataset_list, f, indent= 2)

    print("data saved to: ", os.path.join(save_path, f"{dataset_identifier.replace('/', '-')}.json"))

def extract_and_save_features(dataset, audio_dir, allowed_duration_s= 20):  
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
            
            max_duration_samples = sample_rate * allowed_duration_s
            if len(audio_array) > max_duration_samples:
                audio_array = audio_array[:max_duration_samples]
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

def process_dataset(dataset, split, audio_dir, save_interval = 10_000, total = 460_000):
    all_samples = []
    batch_data = []
    for i, sample in enumerate(tqdm(dataset[split], total= 460_000)):
        batch_data.append({
            "id": i,
            'audio': sample['audio']["array"],
            "sample_rate": sample['audio']["sampling_rate"],
            'true_labels': sample["tags"]
        })
        if len(batch_data) == save_interval:
            batch_dataset = pd.DataFrame(batch_data)      
            batch_data = []
            audio_paths_saved_subset = extract_and_save_features(batch_dataset, audio_dir)  
            all_samples.append(audio_paths_saved_subset)
        
        if i == 400_000:
            break

    if all_samples:
        result_df = pd.concat(all_samples, ignore_index=True)
        print(f"Final dataset size: {len(result_df)} samples")
        return result_df
    else:
        print("No data was loaded")
        return pd.DataFrame()

if __name__ == "__main__":
    dataset_name = "benjamin-paine/freesound-laion-640k"
    split = "train"
    
    dataset_identifier = f"benjamin-paine/freesound-laion-640k_split{split}"
    save_dir = f"/mnt/storage-werent4-2tb/tmp-generic-dataset/r-classes-freesound-laion-640k"
    audio_dir = f"{save_dir}/audio_features_r-classes-freesound-laion-640k_split{split}"

    os.makedirs(save_dir, exist_ok= True)
    os.makedirs(audio_dir, exist_ok= True)


    base_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/"
    data_files = {f"{split}": base_url + f"data/{split}*.parquet"}
    dataset = load_dataset(
        "parquet", 
        data_files=data_files,
        streaming=True
    )
    sampled_subset = process_dataset(dataset, split, audio_dir, save_interval= 1_000)
    labels_sampled_subset = create_label_dataset_sets_scalable_4(sampled_subset, "true_labels", max_additional_labels= 20)
    save_dataset(labels_sampled_subset, save_dir, dataset_identifier)
