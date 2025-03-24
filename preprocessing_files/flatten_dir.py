import os
import shutil

features_root = "/work/users/c/h/chidaksh/MatchTime/features/baidu_soccer_embeddings"

def move_and_flatten_structure(features_root):
    for league_season_dir in os.listdir(features_root):
        league_season_path = os.path.join(features_root, league_season_dir)
        if os.path.isdir(league_season_path):
            for game_folder in os.listdir(league_season_path):
                game_folder_path = os.path.join(league_season_path, game_folder)
                if os.path.isdir(game_folder_path):
                    for nested_dir in os.listdir(game_folder_path):
                        breakpoint()
                        if os.path.isdir(nested_dir):
                            shutil.move(nested_dir, league_season_path)
                            game_folder_path = os.path.join(league_season_path, nested_dir)

if __name__ == "__main__":
    move_and_flatten_structure(features_root)
