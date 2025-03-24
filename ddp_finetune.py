import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from models.matchvoice_model import matchvoice_model
from goal_dataset import Goal_Dataset
from transformers import AdamW
import numpy as np
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider

def eval_cider(predicted_captions, gt_captions):
    cider_evaluator = Cider()
    predicted_captions_dict = {i: [caption] for i, caption in enumerate(predicted_captions)}
    gt_captions_dict = {i: [caption] for i, caption in enumerate(gt_captions)}
    _, cider_scores = cider_evaluator.compute_score(predicted_captions_dict, gt_captions_dict)
    return cider_scores.tolist()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = Goal_Dataset(feature_root=args.feature_root, ann_root=args.train_ann_root, window=args.window, fps=args.fps, tokenizer_name=args.tokenizer_name, timestamp_key=args.train_timestamp_key, stage="mlp")

    val_dataset = Goal_Dataset(feature_root=args.feature_root, ann_root=args.val_ann_root, window=args.window, fps=args.fps, tokenizer_name=args.tokenizer_name, timestamp_key=args.val_timestamp_key, stage="val")

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.train_num_workers, pin_memory=True, collate_fn=train_dataset.collater, drop_last=False)

    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.val_num_workers, pin_memory=True, collate_fn=val_dataset.collater, drop_last=True)

    model = matchvoice_model(llm_ckpt=args.tokenizer_name, tokenizer_ckpt=args.tokenizer_name, window=args.window,num_query_tokens=args.num_query_tokens, num_video_query_token=args.num_video_query_token, num_features=args.num_features, device=device).to(device)
    vocab_size = model.llama_model.config.vocab_size
    print(f"Model vocab size: {vocab_size}")
    print(f"Tokenizer vocab size: {len(model.tokenizer)}")
    assert vocab_size == len(model.tokenizer), "Tokenizer and model vocab mismatch!"

    if args.load_ckpt:
        checkpoint = torch.load(args.load_ckpt, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded checkpoint successfully!")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training"):
            optimizer.zero_grad()
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if 'video_features' in batch:
                batch['features'] = batch.pop('video_features')

            loss = model(samples=batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss}")

        model.eval()
        val_CIDEr = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Validation"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                temp_res_text, anonymized = model(batch, True)
                cur_CIDEr_score = eval_cider(temp_res_text, anonymized)
                val_CIDEr += sum(cur_CIDEr_score) / len(cur_CIDEr_score)
                val_batches += 1
        avg_val_CIDEr = val_CIDEr / val_batches if val_batches > 0 else 0
        print(f"✅ Epoch {epoch+1}: Validation CIDEr = {avg_val_CIDEr}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", type=str, default="./finetune/clip_embeddings/train")
    parser.add_argument("--window", type=float, default=10)
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--train_ann_root", type=str, default="./finetune/videos/train")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--train_num_workers", type=int, default=0)
    parser.add_argument("--train_timestamp_key", type=str, default="gameTime")

    parser.add_argument("--val_ann_root", type=str, default="./finetune/videos/val")
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--val_num_workers", type=int, default=0)
    parser.add_argument("--val_timestamp_key", type=str, default="gameTime")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_query_tokens", type=int, default=32)
    parser.add_argument("--num_video_query_token", type=int, default=32)
    parser.add_argument("--num_features", type=int, default=512)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--model_output_dir", type=str, default="./ckpt/")
    parser.add_argument("--load_ckpt", type=str, default="./ckpt/CLIP_matchvoice.pth")

    args = parser.parse_args()
    train(args)
