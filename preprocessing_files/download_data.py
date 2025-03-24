from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="./")

print("Hi")

# mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["train", "valid", "test"]) # download Features
# mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["train", "valid", "test"]) # download Features reduced with PCA
# mySoccerNetDownloader.downloadGames(files=["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"], split=["train", "valid", "test"]) # download Player Bounding Boxes inferred with MaskRCNN
# mySoccerNetDownloader.downloadGames(files=["1_field_calib_ccbv.json", "2_field_calib_ccbv.json"], split=["train", "valid", "test"]) # download Field Calibration inferred with CCBV
mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=["train", "valid", "test"]) # download Frame Embeddings from https://github.com/baidu-research/vidpress-sports

# print("Going to download games")
# mySoccerNetDownloader.password = "s0cc3rn3t"
# mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test"]) # download 224p Videos
# mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train", "valid", "test"]) # download 720p Videos 
# mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet") # download 720p Videos 
# mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet-Tracking") # download single camera RAW Videos 
