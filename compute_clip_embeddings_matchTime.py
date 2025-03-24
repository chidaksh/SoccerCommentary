import os
import torch
import clip
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

VIDEOS_DIR = "SoccerNetData"
SEC_VIDEOS_DIR = "videos_224p"
SN_ALIGN_DIR = "./dataset/SN-Caption-test-align"
FEATURES_DIR = "features/clip_soccer_embeddings"
FRAME_SIZE = 224
FPS = 2
BATCH_SIZE = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.to(device).eval()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VideoDataset(Dataset):
    def __init__(self, video_path, size=224, fps=2):
        self.video_path = video_path
        self.size = size
        self.fps = fps
        self.transforms = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        
        # Get video metadata once
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video {self.video_path}")

        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        self.frame_indices = [int(x * fps / self.fps) for x in range(int(self.length / fps * self.fps))]

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path)  # Open per frame
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video {self.video_path} during getitem")

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_indices[idx])
        ret, frame = cap.read()
        cap.release()  # Release immediately

        if not ret:
            return torch.zeros(3, self.size, self.size, dtype=torch.float16)  # Return a dummy tensor to avoid issues

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.transforms(Image.fromarray(frame))
        return frame.to(torch.float16)


def encode_features(data_loader, encoder):
    all_features = []
    for frames in tqdm(data_loader, desc="Encoding Frames", leave=False):
        frames = [f for f in frames if f is not None]  # Filter out bad frames
        if not frames:
            continue
        
        frames = torch.stack(frames).to(device).half()  # Move batch to GPU
        with torch.no_grad():
            features = encoder(frames)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0) if all_features else None

def get_relevant_videos():
    relevant_folders = os.listdir(SN_ALIGN_DIR)
    video_folders = []
    video_files = []
    data_dirs = [f"{SN_ALIGN_DIR}/{folder}" for folder in relevant_folders]
    for dir in data_dirs:
        league, year = dir.rsplit("_", 1)
        league = league.rsplit("/", 2)[2]
        for root, dirs, files in os.walk(dir):
            for subdir in dirs:
                video_folders.append(f"{VIDEOS_DIR}/{league}/{year}/{subdir}")
    
    for dir in video_folders:
        found = False
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file in ["1_224p.mkv", "2_224p.mkv"]:
                    found = True
                    video_files.append(os.path.join(root, file))
            if not found:
                _, league, year, game = root.split("/")
                paths = [f"{SEC_VIDEOS_DIR}/{league + '_' + year}/{game}/1_224p.mp4", f"{SEC_VIDEOS_DIR}/{league + year}/{game}/2_224p.mp4"]
                if os.path.isfile(paths[0]):
                    video_files.append(paths[0])
                if os.path.isfile(paths[1]):
                    video_files.append(paths[1])     

    return video_files

def process_video(video_path):
    dataset = VideoDataset(video_path, size=FRAME_SIZE, fps=FPS)
    if len(dataset) == 0:
        print(f"Skipping {video_path}: No valid frames.")
        return

    data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=16, pin_memory=True, prefetch_factor=2
    )

    features = encode_features(data_loader, model.encode_image)
    if features is None:
        print(f"Skipping {video_path}, no features extracted.")
        return

    # breakpoint()
    relative_path = os.path.relpath(video_path, VIDEOS_DIR)
    league, year, dir, vid = relative_path.split("/")
    tail_path = f"{league + '_' + year}/{dir}/{vid.split('_')[0] + '_clip_soccer_embeddings.npy'}"
    save_path = os.path.join(FEATURES_DIR, tail_path)
    ensure_dir(os.path.dirname(save_path))
    np.save(save_path, features)
    print(f"Saved features to {save_path}")


def process_all_videos():
    relevant_videos = get_relevant_videos()
    # breakpoint()
    for video_path in tqdm(relevant_videos, desc="Processing Videos"):
        process_video(video_path)
    print("Feature extraction complete!")

if __name__ == "__main__":
    process_all_videos()
