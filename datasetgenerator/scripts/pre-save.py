import os
import warnings
import json
import torch
import uuid
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset

def add_id(example, idx):
    example['id'] = idx
    return example

def process_labels(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, list):
        return x
    else:
        print("lol 1 label")
        return [x] 

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

if __name__ == "__main__":
    dataset_name = "benjamin-paine/freesound-laion-640k"
    split = "train"
    
    dataset_identifier = f"benjamin-paine/freesound-laion-640k_split{split}"
    save_dir = f"/mnt/storage-werent4-2tb/generic-dataset/r-classes-freesound-laion-640k"
    audio_dir = f"{save_dir}/audio_features_r-classes-freesound-laion-640k_split{split}"

    os.makedirs(save_dir, exist_ok= True)
    os.makedirs(audio_dir, exist_ok= True)
    
    base_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/"
    data_files = {f"{split}": base_url + f"data/{split}*.parquet"}

    dataset = load_dataset("parquet", data_files=data_files,)[split]
    os._exit(-1)
    dataset = dataset.map(add_id, with_indices=True, batch_size= 1000)
    ds_with_path = extract_and_save_features(dataset, audio_dir)
    df_with_path = ds_with_path.to_pandas()
    labeled_ds = create_label_dataset_sets_scalable_4(df_with_path, "tags", min_additional_labels= 2, max_additional_labels= 20)
    save_dataset(labeled_ds, save_dir, dataset_identifier)