import os
import torch
import numpy as np
import clip
import cv2
# from decord import VideoReader, cpu
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Define paths
video_folder = "./finetune/videos/test"
embedding_folder = "./finetune/clip_embeddings/test"
os.makedirs(embedding_folder, exist_ok=True)

# Load CLIP model (ViT-B/32 for speed, ViT-L/14 for accuracy)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model = model.to(device).eval()

class VideoDataset(Dataset):
    def __init__(self, video_path, size=224, fps=2):
        self.video_path = video_path
        self.size = size
        self.fps = fps
        self.transforms = preprocess
        self.frame_indices = self._get_frame_indices()

    def _get_frame_indices(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return [int(x * fps / self.fps) for x in range(int(length / fps * self.fps))]

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path)  # Open video per frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_indices[idx])
        ret, frame = cap.read()
        cap.release()  # Close immediately
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.transforms(Image.fromarray(frame))
        return frame.to(torch.float16)

# Feature Extraction
def encode_features(data_loader, encoder, device):
    # encoder = encoder.to(device)
    all_features = []
    for frames in data_loader:
        frames = frames.to(device)
        with torch.no_grad():
            features = encoder(frames)
        all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)  # Stack features along batch dim

# Process all videos and store embeddings
for file in tqdm(os.listdir(video_folder)):
    if file.endswith(".mp4"):
        match_id = file.replace(".mp4", "")
        video_path = os.path.join(video_folder, file)
        output_path = os.path.join(embedding_folder, f"{match_id}.npy")

        if not os.path.exists(output_path):
            print(f"Processing {match_id}...")
            dataset = VideoDataset(video_path, size=224, fps=2)
            data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
            features = encode_features(data_loader, model.encode_image, device)
            np.save(output_path, features)
            print(f"✅ Saved CLIP embeddings for {match_id}. Shape: {features.shape}")
        else:
            print(f"Skipping {match_id}, embeddings already exist.")

print("🎉 CLIP embeddings computed for all videos!")
