import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models.multi_task import MultiTaskingModel
from configs.util import load_super_config, yaml_to_dict

# ─── 路径配置 ────────────────────────────────────────────────────────────────
CONFIG_PATH      = "./configs/pretrain.yaml"
CHECKPOINT_DIR   = "./pretrained_models/SoccerMaster"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


def load_weights(model: MultiTaskingModel, checkpoint_dir: str):
    model.load_checkpoint(checkpoint_dir, logger=None, load_heads=True)


def make_dummy_inputs(config: dict, device: str):
    """根据 config 构造虚假输入 tensor。"""
    batch_size  = 1
    num_frames  = config["NUM_FRAMES"]
    image_size  = config.get("AUG_MAX_SIZE", 512)
    channels    = 3

    # 视觉输入: [B, T, C, H, W]
    images = torch.randn(batch_size, num_frames, channels, image_size, image_size,
                         device=device)
    return images


def run_inference(model: MultiTaskingModel, images: torch.Tensor,
                  dataset_name: str, device: str):
    model.eval()
    model.to(device)

    with torch.no_grad():
        outputs = model(images, dataset_name=dataset_name)

    return outputs


def print_outputs(outputs: dict):
    print("\n=== 推理输出 ===")
    for head_name, head_output in outputs.items():
        print(f"\n[Head: {head_name}]")
        if isinstance(head_output, dict):
            for k, v in head_output.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, (list, tuple)):
                    print(f"  {k}: list/tuple of length {len(v)}")
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            for kk, vv in item.items():
                                if isinstance(vv, torch.Tensor):
                                    print(f"    [{i}] {kk}: shape={vv.shape}")
                        elif isinstance(item, torch.Tensor):
                            print(f"    [{i}]: shape={item.shape}")
                else:
                    print(f"  {k}: {v}")
        elif isinstance(head_output, torch.Tensor):
            print(f"  tensor shape={head_output.shape}, dtype={head_output.dtype}")
        else:
            print(f"  {head_output}")


def main():
    cfg = yaml_to_dict(CONFIG_PATH)
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))

    model = MultiTaskingModel(config=cfg)
    model.load_checkpoint(CHECKPOINT_DIR, logger=None, load_heads=True)

    images = make_dummy_inputs(cfg, DEVICE)
    print(f"images shape: {images.shape}")  # [B, T, C, H, W]

    # dataset_name = next(iter(cfg["DATASETS_TO_HEADS"]))

    # inference
    model.eval()
    model.to(DEVICE)
    
    for dataset_name in cfg["DATASETS_TO_HEADS"]:
        with torch.no_grad():
            outputs = model(images, dataset_name=dataset_name, text=['dummy text input'])
        print_outputs(outputs)


if __name__ == "__main__":
    main()
