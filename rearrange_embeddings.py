import os
import shutil

# Set the base directory
base_dir = "./InternVideo_features"

for league in os.listdir(base_dir):
    league_path = os.path.join(base_dir, league)
    if os.path.isdir(league_path):
        for season in os.listdir(league_path):
            season_path = os.path.join(league_path, season)
            if os.path.isdir(season_path):
                # Create the new league-season directory
                new_league_season = f"{league}_{season}"
                new_league_season_path = os.path.join(base_dir, new_league_season)
                os.makedirs(new_league_season_path, exist_ok=True) 
                # Move each game folder to the new league-season directory
                for game in os.listdir(season_path):
                    game_path = os.path.join(season_path, game)
                    if os.path.isdir(game_path):
                        new_game_path = os.path.join(new_league_season_path, game)
                        os.makedirs(new_game_path, exist_ok=True)

                        # Rename and move each file
                        for file_name in os.listdir(game_path):
                            if file_name.endswith("_224p_InternVideo.npy"):
                                new_file_name = file_name.replace("_224p_InternVideo.npy", "_internvideo_soccer_embeddings.npy")
                                old_file_path = os.path.join(game_path, file_name)
                                new_file_path = os.path.join(new_game_path, new_file_name)
                                shutil.move(old_file_path, new_file_path)
                        
                        # Remove the old game folder if empty
                        if not os.listdir(game_path):
                            os.rmdir(game_path)

                # Remove the old season folder if empty
                if not os.listdir(season_path):
                    os.rmdir(season_path)

print("Directory restructuring and renaming completed successfully!")