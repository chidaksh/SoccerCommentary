import os
import subprocess
import os
import json
import re
from tqdm import tqdm
import jsonlines as jl
from os import listdir
from os.path import join
import unicodedata
import time

sort_by_length = lambda x: len(x)
metadata_dir = "./storage/football_data/metadata"
metadata_dirs = listdir(metadata_dir)

video_save_path = "./finetune/videos/test"
os.makedirs(video_save_path, exist_ok=True)

def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def convert_game_time(game_time):
    try:
        time_parts = game_time.split(",")[0]
        _, mm, ss = map(int, time_parts.split(":"))
        formatted_time = f"{mm}:{ss:02d}"
        return formatted_time
    except Exception as e:
        print(f"Error converting time: {game_time} -> {e}")
        return game_time

def get_names_for_field(metadata, key, names):
    if type(metadata[key]) == list:
        for item in metadata[key]:
            if item["meta"]:
                names.append(sorted(item["meta"]["nicknames"], key=sort_by_length, reverse=True))
            else:
                names.append([item["name"]])
    elif type(metadata[key]) == dict:
        if metadata[key]["meta"]:
            names.append(sorted(metadata[key]["meta"]["nicknames"], key=sort_by_length, reverse=True))
        else:
            if "name" in metadata[key]:
                names.append([metadata[key]["name"]])

def clean_names(names):
    for _names in list(names):
        for name in reversed(_names):
            if name[0].islower():
                _names.remove(name)

def get_all_names(metadata):
    player_names = []
    coaches_names = []
    team_names = []
    referee_names = []
    stadium_names = []
    if "home_coach" in metadata:
        get_names_for_field(metadata, "home_coach", coaches_names)
    if "away_coach" in metadata:
        get_names_for_field(metadata, "away_coach", coaches_names)
    get_names_for_field(metadata, "home_team", team_names)
    get_names_for_field(metadata, "away_team", team_names)
    get_names_for_field(metadata, "home_lineup", player_names)
    get_names_for_field(metadata, "away_lineup", player_names)
    if "missing_home" in metadata:
        get_names_for_field(metadata, "missing_home", player_names)
    if "missing_away" in metadata:
        get_names_for_field(metadata, "missing_away", player_names)
    if "referee" in metadata["summary"]:
        referee_names.append([metadata["summary"]["referee"]])
    if "stadium" in metadata["summary"]:
        stadium_names.append([metadata["summary"]["stadium"]])
    return player_names, coaches_names, team_names, referee_names, stadium_names

def delexicalize_text(metadata_dirs):
    with jl.open("./storage/football_data/finegrained_splits/test.jsonl") as r:
        ds = []
        for match in tqdm(r):
            if match["match_id"] not in set(metadata_dirs):
                print(f"skipping {match['match_id']}")
                ds.append(match)
                continue
            try:
                metadata = json.load(open(join(metadata_dir, match["match_id"], "match_info.json")))
            except Exception as e:
                print(f"No json file for match {match['match_id']}")
                ds.append(match)
                continue
            players, coaches, teams, referee, stadium = get_all_names(metadata)
            for chunk in match["chunks"]:
                caption = chunk["caption"]
                for names in players:
                    for name in names:
                        normalized_name = normalize_text(name)
                        if name in caption:
                            chunk["caption"] = chunk["caption"].replace(name, "[PLAYER]")
                            if normalized_name in chunk['caption']:
                                chunk["caption"] = chunk["caption"].replace(normalized_name, "[PLAYER]")
                            break
                for names in coaches:
                    for name in names:
                        normalized_name = normalize_text(name)
                        if name in caption:
                            chunk["caption"] = chunk["caption"].replace(name, "[COACH]")
                            if normalized_name in chunk['caption']:
                                chunk["caption"] = chunk["caption"].replace(normalized_name, "[COACH]")
                            break
                for names in teams:
                    for name in names:
                        normalized_name = normalize_text(name)
                        if name in caption:
                            chunk["caption"] = chunk["caption"].replace(name, "[TEAM]")
                            if normalized_name in chunk['caption']:
                                chunk["caption"] = chunk["caption"].replace(normalized_name, "[TEAM]")
                            break
                for names in referee:
                    for name in names:
                        normalized_name = normalize_text(name)
                        if name in caption:
                            chunk["caption"] = chunk["caption"].replace(name, "[REFEREE]")
                            if normalized_name in chunk['caption']:
                                chunk["caption"] = chunk["caption"].replace(normalized_name, "[REFEREE]")
                            break
                for names in stadium:
                    for name in names:
                        normalized_name = normalize_text(name)
                        if name in caption:
                            chunk["caption"] = chunk["caption"].replace(name, "[STADIUM]")
                            if normalized_name in chunk['caption']:
                                chunk["caption"] = chunk["caption"].replace(normalized_name, "[STADIUM]")
                            break
            ds.append(match)
    return ds

def assign_event_label(commentary):
    commentary_lower = commentary.lower()
    if "corner" in commentary_lower:
        return "corner"
    elif any(word in commentary_lower for word in ["goal", "scores", "back of the net"]):
        return "soccer-ball"
    elif any(word in commentary_lower for word in ["yellow card", "booked"]):
        return "y-card"
    elif any(word in commentary_lower for word in ["substitution", "comes on", "off the pitch"]):
        return "substitution"
    elif any(word in commentary_lower for word in ["injury", "medical staff"]):
        return "injury"
    elif any(word in commentary_lower for word in ["kickoff", "full-time", "halftime", "whistle", "half-time"]):
        return "whistle"
    elif any(word in commentary_lower for word in ["penalty awarded", "penalty given", "spot kick", "penalty kick"]):
        return "penalty"
    elif any(word in commentary_lower for word in ["penalty missed", "misses the penalty", "saved penalty"]):
        return "penalty-missed"
    return "" 

def filter_relevant_annotations(annotations):
    return [ann for ann in annotations if len(ann["description"].split()) > 1]

def preprocess_goal_dataset(goal_data, metadata_folder):
    processed_data = []
    count = 0
    delex_captions = delexicalize_text(metadata_dirs)
    for match in goal_data:
        match_id = match["match_id"]
        match_info_path = match.get("match_info", "")

        match_metadata = {}
        if match_info_path:
            try:
                with open(os.path.join(metadata_folder, match_info_path), "r", encoding="utf-8") as f:
                    match_metadata = json.load(f)
            except FileNotFoundError:
                match_metadata = {}

        game_date = match_metadata.get("gameDate", "Unknown")
        venue = match_metadata.get("venue", "Unknown")
        referee = match_metadata.get("referee", [])

        annotations = []
        try:
            captions = delex_captions[count]['chunks']
            count += 1
        except Exception as e:
            print(f"No meta data for match {match['match_id']}")
            annotations.append({})
            count += 1
            continue

        for chunk, caption in zip(match["chunks"], captions):
            event_label = assign_event_label(chunk['caption'])
            annotations.append({
                "gameTime": convert_game_time(chunk["start"]),
                "description": chunk["caption"],
                "anonymized": caption["caption"],
                "label": event_label,
                "start": chunk["start"],
                "end": chunk["end"],
            })
        filtered_annotations = filter_relevant_annotations(annotations)
        processed_data.append({
            "match_id": match_id,
            "timestamp": game_date,
            "venue": venue,
            "referee": referee,
            "annotations": filtered_annotations,
        })
    return processed_data

val_dataset_path = "./storage/football_data/finegrained_splits/test.jsonl"
goal_val_videos = []
goal_objects = []
with open(val_dataset_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        match_data = json.loads(line)
        match_id = match_data.get("match_id", "")
        video_url = match_data.get("URL", "")
        if match_id and video_url:
            goal_val_videos.append({"match_id": match_id, "URL": video_url})
        goal_objects.append(match_data)

processed_data = preprocess_goal_dataset(goal_objects, metadata_folder="./storage/football_data/metadata")
for i, video in tqdm(enumerate(goal_val_videos), total=len(goal_val_videos)):
    match_id = video["match_id"]
    video_url = video["URL"]
    video_path = os.path.join(video_save_path, f"{match_id}.mp4")
    json_path = os.path.join(video_save_path, f"{match_id}.json")
    if i > 295:
        if not os.path.exists(video_path):
            print(f"Downloading {match_id} from {video_url}...")
            cmd = [
                "yt-dlp",
                "--cookies", "/nas/longleaf/home/chidaksh/www.youtube.com_cookies.txt",
                "-o", video_path,
                video_url
            ]
            try:
                subprocess.run(cmd, check=True)
                # time.sleep(2)
            except subprocess.CalledProcessError:
                print(f"Failed to download {match_id}. Skipping...")
                continue

        else:
            print(f"Skipping {match_id}, already downloaded.")

        if i < len(processed_data):
            if match_id == processed_data[i]["match_id"]:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(processed_data[i], f, indent=4)
                print(f"Saved metadata for {match_id}.")
            else:
                print(f"Metadata not found for {match_id}.")
