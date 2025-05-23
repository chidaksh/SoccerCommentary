import os
import random
from torch.utils.data import Dataset
import torch
import numpy as np
import json
from transformers import AutoTokenizer
import copy

IGNORE_INDEX = -100

class Goal_Dataset(Dataset):
    def __init__(self, feature_root, ann_root, window = 15, fps = 2, timestamp_key="gameTime", tokenizer_name = 'meta-llama/Meta-Llama-3-8B', max_token_length=128, stage="combined", val_length=5):
        self.val_length = val_length
        self.stage = stage
        self.caption = traverse_and_parse(ann_root, timestamp_key, self.stage, self.val_length)
        self.feature_root = feature_root
        self.window = window
        self.fps = fps
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = "<|end_of_text|>"
        self.tokenizer.pad_token_id = 128001
        self.tokenizer.add_tokens(["[PLAYER]","[TEAM]","[COACH]","[c]","([TEAM])"], special_tokens=True)
        special_tokens = ["[PLAYER]", "[TEAM]", "[COACH]", "[REFEREE]", "([TEAM])"]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.max_token_length = max_token_length
        self._validate_tokenizer()
        
    def _validate_tokenizer(self):
        try:
            test_text = "[PLAYER] [TEAM] scored a goal!"
            tokens = self.tokenizer.tokenize(test_text)
            assert all(t in tokens for t in ["[PLAYER]", "[TEAM]"])
        except AssertionError:
            missing = [t for t in ["[PLAYER]", "[TEAM]"] if t not in tokens]
            raise ValueError(f"Special tokens {missing} not properly added to tokenizer")


    def __getitem__(self, index):
        num_retries = 50
        fetched_features = None
        # breakpoint()
        for _ in range(num_retries):
            # breakpoint()
            try:
                game, timestamp, label, anonymized = self.caption[index]
                feature_folder = self.feature_root
                file = f"{game}.npy"
                file_paths = [os.path.join(feature_folder, file)]
                fetched_features = torch.from_numpy(load_adjusted_features(file_paths[0], timestamp, self.window, self.fps))
                # anonymized_tokens = self.tokenizer(anonymized, return_tensors = "pt", max_length=self.max_token_length-2,truncation=True, padding='max_length').input_ids[0]
            except:
                breakpoint()
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        return {
            "features": fetched_features.float(),  # Ensure float type
            "anonymized": anonymized,  # Store raw text
            "caption_info": self.caption[index]
        }

    def __len__(self):
        return len(self.caption)
    
    def collater(self, instances):
        text_batch = self.tokenizer(
            [instance["anonymized"] for instance in instances],
            padding='longest',
            max_length=self.max_token_length,
            truncation=True,
            return_tensors='pt'
        )
        
        final_batch = {
            'input_ids': text_batch.input_ids,
            'attention_mask': text_batch.attention_mask,
            'labels': text_batch.input_ids.clone(),
            'caption_info': [instance["caption_info"] for instance in instances]
        }

        final_batch['labels'][final_batch['labels'] == self.tokenizer.pad_token_id] = IGNORE_INDEX
        if 'features' in instances[0]:
            final_batch['features'] = torch.stack([inst['features'] for inst in instances])
        
        return final_batch

def load_adjusted_features(feature_path, timestamp, window, fps=2):
    """
    Load and adjust video features based on the given timestamp and window.

    Args:
    - feature_path (str): The path to the .npy file containing video features.
    - timestamp (int): The target timestamp in seconds.
    - window (float): The window size in seconds.

    Returns:
    - np.array: The adjusted array of video features.
    """
    features = np.load(feature_path)
    total_frames = int(window * 2 * fps)
    if timestamp * fps > len(features):
        return None
    
    start_frame = int(max(0, timestamp - window) * fps + 1)
    end_frame = int((timestamp + window) * fps + 1)
    if end_frame > len(features):
        start_frame = int(max(0, len(features) - total_frames))
    ad = features[start_frame:start_frame+total_frames]
    return ad

def parse_labels_caption(game, file_path, timestamp_key):
    """
    Parses a Labels-caption.json file and extracts the required data.
    Parameters:
        file_path (str): The path to the Labels-caption.json file.
        league (str): The league name.
        game (str): The game name.
    Returns:
        list: A list of tuples containing (half, timestamp, type, anonymized, league, game).
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    result = []
    for annotation in data.get('annotations', []):
        try:
            time = annotation.get(timestamp_key, ' - ')
            # half = int(gameTime.split(' ')[0])
            # if half not in [1, 2]:
            #     continue
            minutes, seconds = map(int, time.split(':'))
            timestamp = minutes * 60 + seconds
            label = annotation.get('label', '')
            anonymized = annotation.get('anonymized', '').replace('[STADIUM]', 'stadium').replace('(TEAM)', "([TEAM])")
            result.append((game, timestamp, label, anonymized))
        except ValueError:
            continue
    return result

def traverse_and_parse(root_dir, timestamp_key, stage, val_len):
    """
    Traverses a directory and its subdirectories to find and parse all Labels-caption.json files.
    Parameters:
        root_dir (str): The root directory to start traversal.
    Returns:
        list: A combined list of tuples from all Labels-caption.json files found.
    """
    all_data = []
    for subdir, dirs, files in os.walk(root_dir):
        if stage == "mlp":
            files=files
        elif stage == "val":
            files=files[:50]
        elif stage == "test":
            files=files[:50]
        else:
            print("Wrong state!")
            exit(-1)
        print(f"Length of videos is {len(files)}")
        for file in files:
            if file.endswith(".json"):
                # league = os.path.basename(os.path.dirname(subdir))
                game = file.split(".")[0]
                file_path = os.path.join(subdir, file)
                all_data.extend(parse_labels_caption(game, file_path, timestamp_key))
    return all_data

