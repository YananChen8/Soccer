from models.deformable_detr.deformable_detr import build_deformable_detr_criterion, build_detection_metrics
from models.lines_detection import build_lines_detection_loss, build_lines_detection_metrics
from models.keypoints_detection import build_keypoints_detection_loss, build_keypoints_detection_metrics
from models.video_caption import build_video_caption_loss, build_video_caption_metrics
from models.caption_classification import build_caption_classification_loss, build_caption_classification_metrics

def build_loss_fn(config: dict):
    loss_fn_dict = {}
    datasets_to_heads = config["DATASETS_TO_HEADS"]
    all_heads = []
    for dataset, heads in datasets_to_heads.items():
        all_heads.extend(heads)
    all_heads = list(set(all_heads))
    all_heads.sort()
    
    for head in all_heads:
        if head == "SoccerNetGSR_Detection":
            loss_fn_dict[head] = build_deformable_detr_criterion(config=config)
        elif head == "LinesDetection":
            loss_fn_dict[head] = build_lines_detection_loss(config=config)
        elif head == "KeypointsDetection":
            loss_fn_dict[head] = build_keypoints_detection_loss(config=config)
        elif head == "VideoCaption":
            loss_fn_dict[head] = build_video_caption_loss(config=config)
        elif head == "CaptionClassification":
            loss_fn_dict[head] = build_caption_classification_loss(config=config)
        else:
            raise ValueError(f"Head {head} is not supported.")

    return loss_fn_dict

def build_metrics_fn(config: dict):
    metrics_fn_dict = {}
    datasets_to_heads = config["DATASETS_TO_HEADS"]
    all_heads = []
    for dataset, heads in datasets_to_heads.items():
        all_heads.extend(heads)
    all_heads = list(set(all_heads))
    all_heads.sort()
    for head in all_heads:
        if head == "SoccerNetGSR_Detection":
            metrics_fn_dict[head] = build_detection_metrics(config=config)
        elif head == "LinesDetection":
            metrics_fn_dict[head] = build_lines_detection_metrics(config=config)
        elif head == "KeypointsDetection":
            metrics_fn_dict[head] = build_keypoints_detection_metrics(config=config)
        elif head == "VideoCaption":
            metrics_fn_dict[head] = build_video_caption_metrics(config=config)
        elif head == "CaptionClassification":
            metrics_fn_dict[head] = build_caption_classification_metrics(config=config)
        else:
            raise ValueError(f"Head {head} is not supported.")

    return metrics_fn_dict