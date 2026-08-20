from data.soccernet_gsr_detection import build_gsr_detection_dataloader
from data.video_caption import build_video_caption_dataloader

def build_dataloader(config: dict, only_test: bool = False):
    dataloader_train_dict = {}
    dataloader_test_dict = {}
    datasets_to_heads = config["DATASETS_TO_HEADS"]
    datasets = datasets_to_heads.keys()
    for dataset in datasets:
        if dataset == "SoccerNetGSR_Detection":
            if not only_test:
                dataloader_train_dict[dataset] = build_gsr_detection_dataloader(config=config, split="train")
            dataloader_test_dict[dataset] = build_gsr_detection_dataloader(config=config, split="test")
        elif dataset == "VideoCaption":
            if not only_test:
                dataloader_train_dict[dataset] = build_video_caption_dataloader(config=config, split="train")
            dataloader_test_dict[dataset] = build_video_caption_dataloader(config=config, split="test")
        else:
            raise ValueError(f"Datasets {datasets} is not supported.")


    return dataloader_train_dict, dataloader_test_dict