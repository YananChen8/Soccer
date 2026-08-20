# ------------------------------------------------------------------------
# Copyright (c) Haolin Yang. All Rights Reserved.
# ------------------------------------------------------------------------
# Evaluation script for trained checkpoints
# ------------------------------------------------------------------------
import torch
import os
import argparse
from datetime import datetime
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import DistributedDataParallelKwargs

from data.build import build_dataloader
from utils.logger import Logger
from models.multi_task import MultiTaskingModel
from utils.misc import set_seed
from configs.util import load_super_config, yaml_to_dict
from models.build import build_loss_fn, build_metrics_fn
from train import evaluate_one_epoch

def save_results_with_logger(eval_results: dict, logger: Logger):
    """
    Log evaluation results to the logger.

    Args:
        eval_results: Dict containing evaluation results.
        logger: Logger instance.
    """
    logger.info("=== Evaluation Results Summary ===")
    
    if "head_metrics_results" in eval_results:
        logger.info("--- Head Metrics Results ---")
        head_metrics = eval_results["head_metrics_results"]
        
        if head_metrics:
            for head_name, metrics_dict in head_metrics.items():
                if metrics_dict:
                    logger.info(f"{head_name} Metrics:")
                    for metric_name, metric_value in metrics_dict.items():
                        logger.info(f"  {metric_name}: {metric_value:.4f}")
                else:
                    logger.info(f"{head_name}: No metrics available")
        else:
            logger.info("No head metrics results available")
    
    logger.info("=== End of Evaluation Results ===")

def evaluation_engine(config: dict, checkpoint_path: str, log_dir: str = None):
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)]
    )
    state = PartialState()
    
    set_seed(config["SEED"])
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = os.path.basename(checkpoint_path.rstrip('/'))
        log_dir = f"eval_logs_{checkpoint_name}_{timestamp}"
    
    logger = Logger(
        log_dir=log_dir,
        accelerator=accelerator,
        config=config,
        use_tensorboard=False,
        tensorboard_flush_secs=30
    )
    
    _, dataloader_test_dict = build_dataloader(config=config, only_test=True)
    dataloader_test_dict = {task: dataloader for task, dataloader in dataloader_test_dict.items() 
                           if dataloader is not None}
    
    if not dataloader_test_dict:
        raise ValueError("No test datasets available for evaluation!")
    
    logger.info(f"Test datasets available for tasks: {list(dataloader_test_dict.keys())}")
    
    loss_fn_dict = build_loss_fn(config=config)
    metrics_fn_dict = build_metrics_fn(config=config)
    
    if config["MODEL_ARCH"] == "multitask":
        model = MultiTaskingModel(config=config, logger=logger)
    else:
        raise ValueError(f"Invalid model architecture: {config['MODEL_ARCH']}")
    
    if config["MODEL_ARCH"] == "multitask":
        model.load_checkpoint(checkpoint_path, logger, load_heads=True)
    
    model = accelerator.prepare(model)
    dataloader_test_dict = {task: accelerator.prepare(dataloader) for task, dataloader in dataloader_test_dict.items()}
    
    if config["MODEL_ARCH"] == "multitask":
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100
        
        original_model = model.module if hasattr(model, 'module') else model
        vision_params = sum(p.numel() for p in original_model.backbone.vision_model.parameters())
        vision_trainable_params = sum(p.numel() for p in original_model.backbone.vision_model.parameters() if p.requires_grad)
        
        head_params = {}
        head_trainable_params = {}
        total_head_params = 0
        total_head_trainable_params = 0
        
        for head_name, head in original_model.multi_task_head.items():
            head_total = sum(p.numel() for p in head.parameters())
            head_train = sum(p.numel() for p in head.parameters() if p.requires_grad)
            head_params[head_name] = head_total
            head_trainable_params[head_name] = head_train
            total_head_params += head_total
            total_head_trainable_params += head_train
        
        logger.info(f"=== Model Parameter Statistics (Unit: M) ===")
        logger.info(f"Total parameters: {total_params/1e6:.2f}M")
        logger.info(f"Trainable parameters: {trainable_params/1e6:.2f}M ({trainable_percentage:.2f}%)")
        logger.info(f"Non-trainable parameters: {(total_params - trainable_params)/1e6:.2f}M ({100 - trainable_percentage:.2f}%)")
        logger.info(f"")
        logger.info(f"Vision Model parameters: {vision_params/1e6:.2f}M")
        logger.info(f"Vision Model trainable: {vision_trainable_params/1e6:.2f}M ({vision_trainable_params/vision_params*100:.2f}%)")
        logger.info(f"")
        logger.info(f"Total Head parameters: {total_head_params/1e6:.2f}M")
        logger.info(f"Total Head trainable: {total_head_trainable_params/1e6:.2f}M ({total_head_trainable_params/total_head_params*100:.2f}%)")
        logger.info(f"")
        for head_name in head_params:
            logger.info(f"{head_name} Head parameters: {head_params[head_name]/1e6:.2f}M")
            logger.info(f"{head_name} Head trainable: {head_trainable_params[head_name]/1e6:.2f}M ({head_trainable_params[head_name]/head_params[head_name]*100:.2f}%)")
        logger.info(f"============================================")
    
    logger.info("Starting evaluation...")
    eval_results = evaluate_one_epoch(
        config=config,
        accelerator=accelerator,
        epoch=0,
        dataloader_dict=dataloader_test_dict,
        loss_fn_dict=loss_fn_dict,
        metrics_fn_dict=metrics_fn_dict,
        model=model,
        logger=logger,
    )
    logger.info("Evaluation completed!")
    
    if accelerator.is_main_process:
        save_results_with_logger(eval_results, logger)
    
    accelerator.wait_for_everyone()
    
    return eval_results


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model checkpoint')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file or directory')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Log directory path for results (default: auto-generated)')
    parser.add_argument('--super_config', type=str, default=None,
                       help='Path to super config file')
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    args = parse_args()
    cfg = yaml_to_dict(args.config)
    
    if args.super_config is not None:
        cfg = load_super_config(cfg, args.super_config)
    else:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    
    cfg["CHECKPOINT_PATH"] = args.checkpoint
    if args.log_dir:
        cfg["LOG_DIR"] = args.log_dir
    
    try:
        eval_results = evaluation_engine(
            config=cfg,
            checkpoint_path=args.checkpoint,
            log_dir=args.log_dir,
        )
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1) 