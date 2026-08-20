# ------------------------------------------------------------------------
# Copyright (c) Haolin Yang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
import os
os.environ["NCCL_TIMEOUT"] = "7200"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from torch.utils.data import DataLoader
import torch.nn as nn
from datetime import timedelta
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
import json
import glob

from data.build import build_dataloader
from utils.logger import Logger, MetricsTracker, TPS, Metrics
from models.multi_task import MultiTaskingModel
from runtime_option import runtime_option
from utils.misc import is_distributed, set_seed
from configs.util import load_super_config, update_config, yaml_to_dict
from models.build import build_loss_fn, build_metrics_fn
import math

class CosineAnnealingLRWithWarmup(_LRScheduler):
    """
    Cosine Annealing LR scheduler that starts after warmup epochs.
    During warmup epochs, this scheduler does nothing (lr is controlled by warmup function).
    After warmup, it applies cosine annealing from the base lr to min_lr.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-8, last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer
            warmup_epochs: Number of warmup epochs (scheduler starts after this)
            total_epochs: Total number of training epochs
            min_lr: Minimum learning rate (can be a single value or list for each param group)
            last_epoch: The index of last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.cosine_epochs = total_epochs - warmup_epochs
        
        if not isinstance(min_lr, (list, tuple)):
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        else:
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        
        super(CosineAnnealingLRWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        cosine_epoch = self.last_epoch - self.warmup_epochs
        progress = cosine_epoch / self.cosine_epochs
        
        return [
            min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]

def find_latest_checkpoint(outputs_dir):
    """
    Find the latest checkpoint directory in outputs_dir
    
    Args:
        outputs_dir: Output directory path
        
    Returns:
        tuple: (latest_epoch_dir, epoch_number) or (None, None) if no checkpoint found
    """
    if not os.path.exists(outputs_dir):
        return None, None
    
    epoch_dirs = glob.glob(os.path.join(outputs_dir, "epoch_*"))
    if not epoch_dirs:
        return None, None
    
    valid_epochs = []
    for epoch_dir in epoch_dirs:
        try:
            epoch_num = int(os.path.basename(epoch_dir).split("_")[1])
            # Check if this epoch has a complete checkpoint (training_state.json should exist)
            training_state_file = os.path.join(epoch_dir, "training_state.json")
            if os.path.exists(training_state_file):
                valid_epochs.append((epoch_num, epoch_dir))
        except (ValueError, IndexError):
            continue
    
    if not valid_epochs:
        return None, None
    
    valid_epochs.sort(key=lambda x: x[0])
    latest_epoch, latest_dir = valid_epochs[-1]
    
    return latest_dir, latest_epoch

def save_training_state(outputs_dir, epoch, model, optimizer, scheduler, train_states, accelerator, logger):
    """
    Save complete training state including model, optimizer, scheduler and training progress
    
    Args:
        outputs_dir: Output directory
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        train_states: Training states dictionary
        accelerator: Accelerator instance
        logger: Logger instance
    """
    # Use model.module to access original model attributes when using DDP
    original_model = model.module if hasattr(model, 'module') else model
    
    epoch_dir = os.path.join(outputs_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    original_model.save_checkpoint(epoch_dir, logger)
    
    optimizer_state_file = os.path.join(epoch_dir, "optimizer_state.pt")
    torch.save(optimizer.state_dict(), optimizer_state_file)
    
    scheduler_state_file = os.path.join(epoch_dir, "scheduler_state.pt")
    torch.save(scheduler.state_dict(), scheduler_state_file)
    
    training_state = {
        "epoch": epoch,
        "global_step": train_states["global_step"],
        "start_epoch": train_states["start_epoch"],
        "random_states": {
            "python": torch.get_rng_state().tolist(),
            "numpy": torch.random.get_rng_state().tolist() if hasattr(torch.random, 'get_rng_state') else None,
            "cuda": torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else None,
        }
    }
    
    training_state_file = os.path.join(epoch_dir, "training_state.json")
    with open(training_state_file, 'w') as f:
        json.dump(training_state, f, indent=2)
    
    logger.info(f"Saved complete training state for epoch {epoch} to {epoch_dir}")

def load_training_state(checkpoint_dir, ckpt_type, model, optimizer, scheduler, train_states, accelerator, logger):
    """
    Load complete training state from checkpoint
    
    Args:
        checkpoint_dir: Checkpoint directory path
        ckpt_type: Checkpoint type
        model: Model to load state into
        optimizer: Optimizer to load state into  
        scheduler: Learning rate scheduler to load state into
        train_states: Training states dictionary to update
        accelerator: Accelerator instance
        logger: Logger instance
        
    Returns:
        int: Loaded epoch number
    """
    # Use model.module to access original model attributes when using DDP
    original_model = model.module if hasattr(model, 'module') else model
    
    logger.info(f"Loading training state from {checkpoint_dir}")
    
    original_model.load_checkpoint(checkpoint_dir, ckpt_type, logger)
    
    optimizer_state_file = os.path.join(checkpoint_dir, "optimizer_state.pt")
    optimizer.load_state_dict(torch.load(optimizer_state_file, map_location='cpu'))
    logger.info(f"Loaded optimizer state from {optimizer_state_file}")
    
    scheduler_state_file = os.path.join(checkpoint_dir, "scheduler_state.pt")
    scheduler.load_state_dict(torch.load(scheduler_state_file, map_location='cpu'))
    logger.info(f"Loaded scheduler state from {scheduler_state_file}")
    
    training_state_file = os.path.join(checkpoint_dir, "training_state.json")
    resumed_epoch = 0
    with open(training_state_file, 'r') as f:
        training_state = json.load(f)
    
    resumed_epoch = training_state["epoch"]
    train_states["global_step"] = training_state["global_step"]
    train_states["start_epoch"] = training_state["start_epoch"]
    
    # Restore random states for reproducibility
    if "random_states" in training_state:
        random_states = training_state["random_states"]
        if random_states["python"]:
            torch.set_rng_state(torch.tensor(random_states["python"], dtype=torch.uint8))
        if random_states["numpy"] and hasattr(torch.random, 'set_rng_state'):
            torch.random.set_rng_state(torch.tensor(random_states["numpy"], dtype=torch.uint8))
        if random_states["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state(torch.tensor(random_states["cuda"], dtype=torch.uint8))
    
    logger.info(f"Loaded training state: epoch={resumed_epoch}, global_step={train_states['global_step']}")
    
    return resumed_epoch

# Create parameter groups with different learning rates
def create_param_groups(model, config):
    original_model = model.module if hasattr(model, 'module') else model
    
    param_groups = []
    
    backbone_params = []
    temporal_embedding_params = []
    other_temporal_params = []
    
    for name, param in original_model.backbone.vision_model.named_parameters():
        if param.requires_grad:
            if "temporal_embedding" in name:
                temporal_embedding_params.append(param)
            elif ('temporal' in name and 'embedding' not in name):
                other_temporal_params.append(param)
            else:
                backbone_params.append(param)
    
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': config["LR_BACKBONE"],
            'weight_decay': config["WEIGHT_DECAY"],
            'name': 'backbone'
        })
    
    # Add temporal_embedding parameters with conditional weight decay
    if temporal_embedding_params:
        temporal_embedding_weight_decay = 0.0 if config["EXCLUDE_TEMPORAL_EMBEDDING_WEIGHT_DECAY"] else config["WEIGHT_DECAY"]
        
        param_groups.append({
            'params': temporal_embedding_params,
            'lr': config["LR_BACKBONE_TEMPORAL"],
            'weight_decay': temporal_embedding_weight_decay,
            'name': 'backbone_temporal_embedding'
        })
    
    # Add other temporal parameters with normal weight decay
    if other_temporal_params:
        param_groups.append({
            'params': other_temporal_params,
            'lr': config["LR_BACKBONE_TEMPORAL"],
            'weight_decay': config["WEIGHT_DECAY"],
            'name': 'backbone_temporal_other'
        })
    
    text_encoder_params = []
    if original_model.backbone.text_model is not None:
        for name, param in original_model.backbone.text_model.named_parameters():
            if param.requires_grad:
                text_encoder_params.append(param)
    
    if text_encoder_params:
        param_groups.append({
            'params': text_encoder_params,
            'lr': config["LR_TEXT_ENCODER"],
            'weight_decay': config["WEIGHT_DECAY"],
            'name': 'text_encoder'
        })
    
    # Head parameters with different learning rates
    head_lr_mapping = {
        'SoccerNetGSR_Detection': config["LR_SOCCERNET_GSR_DETECTION"],
        'LinesDetection': config["LR_LINES_DETECTION"],
        'KeypointsDetection': config["LR_KEYPOINTS_DETECTION"],
        'VideoCaption': config["LR_VIDEO_CAPTION"],
        'CaptionClassification': config["LR_CAPTION_CLASSIFICATION"],
    }
    
    for head_name, head in original_model.multi_task_head.items():
        if head_name in ['CaptionClassification']:
            # Separate classifier parameters from other parameters
            classifier_params = []
            other_params = []
            
            for name, param in head.named_parameters():
                if param.requires_grad:
                    if 'classifier' in name.lower():
                        classifier_params.append(param)
                    else:
                        other_params.append(param)
            
            if classifier_params:
                param_groups.append({
                    'params': classifier_params,
                    'lr': config["LR_CAPTION_CLASSIFICATION_CLASSIFIER"],
                    'weight_decay': config["WEIGHT_DECAY"],
                    'name': f'{head_name}_classifier'
                })
            
            if other_params:
                param_groups.append({
                    'params': other_params,
                    'lr': head_lr_mapping[head_name],
                    'weight_decay': config["WEIGHT_DECAY"],
                    'name': f'{head_name}_other'
                })
        else:
            head_params = []
            for param in head.parameters():
                if param.requires_grad:
                    head_params.append(param)
            
            if head_params:
                lr = head_lr_mapping[head_name]
                param_groups.append({
                    'params': head_params,
                    'lr': lr,
                    'weight_decay': config["WEIGHT_DECAY"],
                    'name': head_name
                })
    
    return param_groups

def train_engine(config: dict):
    assert "EXP_NAME" in config and config["EXP_NAME"] is not None, "Please set the experiment name."
    outputs_dir = config["OUTPUTS_DIR"] if config["OUTPUTS_DIR"] is not None \
        else os.path.join("./outputs/", config["EXP_NAME"])

    accelerator = Accelerator(
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False),
            InitProcessGroupKwargs(timeout=timedelta(minutes=120))
        ]
    )
    state = PartialState()
    set_seed(config["SEED"])
    
    log_dir = os.path.join(outputs_dir, "logs")
    logger = Logger(
        log_dir=log_dir,
        accelerator=accelerator,
        config=config,
        use_tensorboard=config["USE_TENSORBOARD"],
        tensorboard_flush_secs=config["TENSORBOARD_FLUSH_SECS"]
    )
    logger.config(config=config)
    
    dataloader_train_dict, dataloader_test_dict = build_dataloader(config=config)
    
    # Filter out None test dataloaders (some tasks might not have test sets)
    dataloader_test_dict = {dataset: dataloader for dataset, dataloader in dataloader_test_dict.items() 
                           if dataloader is not None}
    
    if dataloader_test_dict:
        logger.info(f"Test datasets available for tasks: {list(dataloader_test_dict.keys())}")
    else:
        logger.warning("No test datasets available. Evaluation will be skipped.")
    
    loss_fn_dict = build_loss_fn(config=config)
    metrics_fn_dict = build_metrics_fn(config=config)
    
    if config["MODEL_ARCH"] == "multitask":
        model = MultiTaskingModel(config=config, logger=logger)
    else:
        raise ValueError(f"Invalid model architecture: {config['MODEL_ARCH']}")
    
    param_groups = create_param_groups(model, config)
    optimizer = AdamW(param_groups)
    
    scheduler_type = config["SCHEDULER_TYPE"]
    if scheduler_type == "MultiStepLR":
        scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=config["SCHEDULER_MILESTONES"],
            gamma=config["SCHEDULER_GAMMA"],
        )
        logger.info(f"Using MultiStepLR scheduler with milestones={config['SCHEDULER_MILESTONES']}, gamma={config['SCHEDULER_GAMMA']}")
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLRWithWarmup(
            optimizer=optimizer,
            warmup_epochs=config["LR_WARMUP_EPOCHS"],
            total_epochs=config["EPOCHS"],
            min_lr=config["SCHEDULER_MIN_LR"],
        )
        logger.info(f"Using CosineAnnealingLR scheduler with warmup_epochs={config['LR_WARMUP_EPOCHS']}, "
                   f"total_epochs={config['EPOCHS']}, min_lr={config['SCHEDULER_MIN_LR']}")
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Supported types: 'MultiStepLR', 'CosineAnnealingLR'")
    
    train_states = {
        "start_epoch": 0,
        "global_step": 0
    }
    
    resume_from_checkpoint = False
    resume_epoch = 0
    if config["RESUME_TRAINING"]:
        checkpoint_dir, latest_epoch = find_latest_checkpoint(outputs_dir)
        if checkpoint_dir is not None:
            logger.info(f"Found checkpoint at epoch {latest_epoch}, will resume training from {checkpoint_dir}")
            resume_from_checkpoint = True
            ckpt_type = 'soccer_master'
            # We'll load the state after preparing model, optimizer, scheduler
        else:
            logger.info("RESUME_TRAINING is True but no valid checkpoint found, starting from scratch")
    
    # Alternative: resume from specific checkpoint directory
    if config["RESUME_FROM_CHECKPOINT_DIR"] is not None:
        resume_checkpoint_dir = config["RESUME_FROM_CHECKPOINT_DIR"]
        if os.path.exists(resume_checkpoint_dir):
            training_state_file = os.path.join(resume_checkpoint_dir, "training_state.json")
            if os.path.exists(training_state_file):
                logger.info(f"Will resume training from specified checkpoint: {resume_checkpoint_dir}")
                resume_from_checkpoint = True
                checkpoint_dir = resume_checkpoint_dir
                ckpt_type = 'soccer_master'
            else:
                logger.warning(f"Specified checkpoint directory {resume_checkpoint_dir} does not contain valid training state")
        else:
            logger.warning(f"Specified checkpoint directory {resume_checkpoint_dir} does not exist")

    if resume_from_checkpoint:
        resume_epoch = load_training_state(
            checkpoint_dir, ckpt_type, model, optimizer, scheduler, train_states, accelerator, logger
        )
        # Update start_epoch to resume from the next epoch
        train_states["start_epoch"] = resume_epoch + 1
        logger.info(f"Resuming training from epoch {resume_epoch + 1}, global_step {train_states['global_step']}")
        
        # Mark resume point in tensorboard
        logger.mark_resume(resume_epoch, train_states['global_step'])
            
    model, optimizer = accelerator.prepare(model, optimizer)
    dataloader_train_dict = {dataset: accelerator.prepare(dataloader) for dataset, dataloader in dataloader_train_dict.items()}
    if dataloader_test_dict:
        dataloader_test_dict = {dataset: accelerator.prepare(dataloader) for dataset, dataloader in dataloader_test_dict.items()}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params) * 100
    
    original_model = model.module if hasattr(model, 'module') else model
    vision_params = sum(p.numel() for p in original_model.backbone.vision_model.parameters())
    vision_trainable_params = sum(p.numel() for p in original_model.backbone.vision_model.parameters() if p.requires_grad)
    
    if original_model.backbone.text_model is not None:
        text_params = sum(p.numel() for p in original_model.backbone.text_model.parameters())
        text_trainable_params = sum(p.numel() for p in original_model.backbone.text_model.parameters() if p.requires_grad)
    
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
    if original_model.backbone.text_model is not None:
        logger.info(f"Text Model parameters: {text_params/1e6:.2f}M")
        logger.info(f"Text Model trainable: {text_trainable_params/1e6:.2f}M ({text_trainable_params/text_params*100:.2f}%)")
        logger.info(f"")
    logger.info(f"Total Head parameters: {total_head_params/1e6:.2f}M")
    logger.info(f"Total Head trainable: {total_head_trainable_params/1e6:.2f}M ({total_head_trainable_params/total_head_params*100:.2f}%)")
    logger.info(f"")
    for head_name in head_params:
        logger.info(f"{head_name} Head parameters: {head_params[head_name]/1e6:.2f}M")
        logger.info(f"{head_name} Head trainable: {head_trainable_params[head_name]/1e6:.2f}M ({head_trainable_params[head_name]/head_params[head_name]*100:.2f}%)")
    logger.info(f"============================================")
    
    logger.info(f"=== Learning Rate Configuration ===")
    for i, param_group in enumerate(optimizer.param_groups):
        group_name = param_group['name']
        group_lr = param_group['lr']
        group_params = len(param_group['params'])
        logger.info(f"{group_name}: LR={group_lr:.0e}, Parameters={group_params}")
    logger.info(f"=====================================")
    
    for epoch in range(train_states["start_epoch"], config["EPOCHS"]):
        train_one_epoch(
            config=config,
            accelerator=accelerator,
            states=train_states,
            epoch=epoch,
            dataloader_dict=dataloader_train_dict,
            loss_fn_dict=loss_fn_dict,
            model=model,
            optimizer=optimizer,
            logger=logger,
            lr_warmup_epochs=config["LR_WARMUP_EPOCHS"],
            accumulate_steps=config["ACCUMULATE_STEPS"],
            max_clip_norm=config["MAX_CLIP_NORM"],
            use_accelerate_clip_norm=config["USE_ACCELERATE_CLIP_NORM"],
            use_clip_grad_norm=config["USE_CLIP_GRAD_NORM"],
            logging_interval=config["LOGGING_INTERVAL"],
        )
        scheduler.step()
        if is_distributed():
            torch.distributed.barrier()
        
        # Evaluate after each epoch if test datasets are available
        if dataloader_test_dict and (epoch + 1) % config["EVAL_PER_EPOCH"] == 0:
            logger.info(f"Starting evaluation for epoch {epoch}...")
            eval_results = evaluate_one_epoch(
                config=config,
                accelerator=accelerator,
                epoch=epoch,
                dataloader_dict=dataloader_test_dict,
                loss_fn_dict=loss_fn_dict,
                metrics_fn_dict=metrics_fn_dict,
                model=model,
                logger=logger
            )
            logger.info(f"Evaluation completed for epoch {epoch}")
        if is_distributed():
            torch.distributed.barrier()
        
        if (epoch + 1) % config["SAVE_CHECKPOINT_PER_EPOCH"] == 0:
            save_training_state(
                outputs_dir, epoch, model, optimizer, scheduler, train_states, accelerator, logger
            )
    
    if logger:
        logger.close_tb_writer()
        logger.info("Training completed. TensorBoard logger closed.")

def evaluate_one_epoch(
    config: dict,
    accelerator: Accelerator,
    epoch: int,
    dataloader_dict: dict[str, DataLoader],
    loss_fn_dict: dict[str, nn.Module],
    metrics_fn_dict: dict[str, nn.Module],
    model,
    logger: Logger,
):
    """
    Evaluate model on test dataset for one epoch and log results to tensorboard
    
    Args:
        config: Configuration dictionary
        accelerator: Accelerator instance
        epoch: Current epoch number
        dataloader_dict: Dictionary of test dataloaders for each dataset
        loss_fn_dict: Dictionary of loss functions for each head
        metrics_fn_dict: Dictionary of metrics functions for each head
        model: Model to evaluate
        logger: Logger instance
        
    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    device = accelerator.device
    
    datasets_to_heads = config["DATASETS_TO_HEADS"]
    all_heads = []
    for dataset_name, heads in datasets_to_heads.items():
        all_heads.extend(heads)
    all_heads = list(set(all_heads))
    all_heads.sort()
    
    eval_weighted_losses = {head: MetricsTracker() for head in all_heads}
    eval_unweighted_losses = {head: MetricsTracker() for head in all_heads}
    eval_log_only_losses = {head: MetricsTracker() for head in all_heads}
    
    head_sample_counts = {head: 0 for head in all_heads}
    
    for _name, _loader in dataloader_dict.items():
        sampler = getattr(_loader, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'set_epoch'):
            try:
                sampler.set_epoch(epoch)
            except Exception:
                pass

    for dataset_name, dataloader in dataloader_dict.items():
        logger.info(f"Evaluating dataset: {dataset_name}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images, annotations, metas = batch.values()
                if type(images) == torch.Tensor:
                    batch_size = images.size(0)
                elif type(images) == list or type(images) == tuple:
                    batch_size = len(images)
                else:
                    raise ValueError(f"Unknown image type: {type(images)}")
                
                if dataset_name in ["VideoCaption"]:
                    text = [annotation['text'] for annotation in annotations]
                else:
                    text = None
                
                with accelerator.autocast():
                    outputs = model(images, dataset_name, metas, text)
                    
                    for head_name in datasets_to_heads[dataset_name]:
                        if head_name == "VideoCaption":
                            loss_output = loss_fn_dict[head_name](outputs[head_name], annotations, metas)
                        else:
                            loss_output = loss_fn_dict[head_name](outputs[head_name], annotations)
                        
                        if head_name in ["SoccerNetGSR_Detection"]:
                            loss_task_raw, weight_dict, _ = loss_output
                            
                            weighted_losses = {k: (v * weight_dict[k]) for k, v in loss_task_raw.items() if k in weight_dict}
                            unweighted_losses = {k: v for k, v in loss_task_raw.items() if k in weight_dict}
                            log_only_losses = {k: v for k, v in loss_task_raw.items() if k not in weight_dict}
                            eval_weighted_losses[head_name].update(weighted_losses)
                            eval_unweighted_losses[head_name].update(unweighted_losses)
                            eval_log_only_losses[head_name].update(log_only_losses)
                            if metrics_fn_dict[head_name] is not None:
                                if 'target_sizes' in metas:
                                    target_sizes = metas['target_sizes']
                                else:
                                    target_sizes = torch.tensor([[512, 512]] * batch_size, device=device)
                                metrics_fn_dict[head_name].update(outputs[head_name], annotations, target_sizes)
                                
                        elif head_name in ["VideoCaption", "CaptionClassification", "LinesDetection", "KeypointsDetection"]:
                            loss_task_raw, weight_dict = loss_output
                            
                            weighted_losses = {k: (v * weight_dict[k]) for k, v in loss_task_raw.items() if k in weight_dict}
                            unweighted_losses = {k: v for k, v in loss_task_raw.items() if k in weight_dict}
                            log_only_losses = {k: v for k, v in loss_task_raw.items() if k not in weight_dict}
                            eval_weighted_losses[head_name].update(weighted_losses)
                            eval_unweighted_losses[head_name].update(unweighted_losses)
                            eval_log_only_losses[head_name].update(log_only_losses)
                            if metrics_fn_dict[head_name] is not None:
                                if head_name == "VideoCaption":
                                    metrics_fn_dict[head_name].update(outputs[head_name], annotations, loss_task_raw)
                                else:
                                    metrics_fn_dict[head_name].update(outputs[head_name], annotations)
                                
                        else:
                            raise ValueError(f"Unknown head name: {head_name}")
                        
                        head_sample_counts[head_name] += batch_size
        
        logger.info(f"Dataset {dataset_name} eval completed")
    
    final_metrics_results = {}
    for head_name in all_heads:
        if metrics_fn_dict[head_name] is not None:
            if is_distributed():
                torch.distributed.barrier()
            final_metrics = metrics_fn_dict[head_name].compute_final_metrics(accelerator)
            if accelerator.is_main_process:
                final_metrics_results[head_name] = final_metrics
            else:
                final_metrics_results[head_name] = {}
            metrics_fn_dict[head_name].reset()
    
    if logger:
        total_weighted_loss = 0.0
        total_unweighted_loss = 0.0
        total_samples = sum(head_sample_counts.values())
        
        for head_name in all_heads:
            head_weighted_avg = eval_weighted_losses[head_name].get_averages()
            head_unweighted_avg = eval_unweighted_losses[head_name].get_averages()
            head_log_only_avg = eval_log_only_losses[head_name].get_averages()
            
            head_weighted_total = sum(head_weighted_avg.values()) if head_weighted_avg else 0.0
            head_unweighted_total = sum(head_unweighted_avg.values()) if head_unweighted_avg else 0.0
            
            sample_weight = head_sample_counts[head_name] / total_samples if total_samples > 0 else 0.0
            total_weighted_loss += head_weighted_total * sample_weight
            total_unweighted_loss += head_unweighted_total * sample_weight
            
            if head_weighted_avg:
                for metric_name, value in head_weighted_avg.items():
                    logger.log_scalar(f"eval_weighted_{head_name}/{metric_name}", value, epoch)
                logger.log_scalar(f"eval_weighted_{head_name}/total_loss", head_weighted_total, epoch)
            
            if head_unweighted_avg:
                for metric_name, value in head_unweighted_avg.items():
                    logger.log_scalar(f"eval_unweighted_{head_name}/{metric_name}", value, epoch)
                logger.log_scalar(f"eval_unweighted_{head_name}/total_loss", head_unweighted_total, epoch)
            
            if head_log_only_avg:
                for metric_name, value in head_log_only_avg.items():
                    logger.log_scalar(f"eval_unweighted_{head_name}/{metric_name}", value, epoch)
            
            if head_name in final_metrics_results:
                for metric_name, value in final_metrics_results[head_name].items():
                    logger.log_scalar(f"eval_metrics_{head_name}/{metric_name}", value, epoch)
        
        logger.log_scalar("eval_overall/weighted_total_loss", total_weighted_loss, epoch)
        logger.log_scalar("eval_overall/unweighted_total_loss", total_unweighted_loss, epoch)
        logger.log_scalar("eval_overall/total_samples", total_samples, epoch)
        
        logger.flush_tb_writer()
    
    results = {
        "head_weighted_results": {head: eval_weighted_losses[head].get_averages() for head in all_heads},
        "head_unweighted_results": {head: eval_unweighted_losses[head].get_averages() for head in all_heads},
        "head_log_only_results": {head: eval_log_only_losses[head].get_averages() for head in all_heads},
        "head_metrics_results": final_metrics_results,
        "head_sample_counts": head_sample_counts,
        "overall_weighted_loss": total_weighted_loss if logger else 0.0,
        "overall_unweighted_loss": total_unweighted_loss if logger else 0.0
    }
    
    return results
    
def train_one_epoch(
        config: dict,
        accelerator: Accelerator,
        states: dict,
        epoch: int,
        dataloader_dict: dict[str, DataLoader],
        loss_fn_dict: dict[str, nn.Module],
        model,
        optimizer,
        logger: Logger,
        lr_warmup_epochs: int = 0,
        accumulate_steps: int = 1,
        max_clip_norm: float = 0.1,
        use_accelerate_clip_norm: bool = True,
        use_clip_grad_norm: bool = True,
        logging_interval: int = 20,
):
    epoch_start_timestamp = TPS.timestamp()
    current_last_checkpoint_idx = 0
    model.train()
    tps = TPS()
    metrics = Metrics()
    step_timestamp = tps.timestamp()
    optimizer.zero_grad()
    device = accelerator.device
    
    assert accumulate_steps == 1, "accumulate_steps must be 1 for now."
    
    datasets_to_heads = config["DATASETS_TO_HEADS"]
    all_heads = []
    for dataset_name, heads in datasets_to_heads.items():
        all_heads.extend(heads)
    all_heads = list(set(all_heads))
    all_heads.sort()
    
    max_iterations = max(len(dataloader) for dataloader in dataloader_dict.values())
    
    epoch_metrics = MetricsTracker()
    
    logger.info(f"Training epoch {epoch} with {max_iterations} iterations...")
    
    dataloader_lengths = {task: len(dataloader) for task, dataloader in dataloader_dict.items()}
    logger.info(f"Dataloader lengths: {dataloader_lengths}")

    dataloader_iters = {task: iter(dataloader) for task, dataloader in dataloader_dict.items()}
    
    len_tasks = len(dataloader_iters)

    for cur_iter in range(max_iterations):
        weighted_loss_dict = {}
        unweighted_loss_dict = {}
        log_only_loss_dict = {}
        
        for dataset_name, dataloader_iter in dataloader_iters.items():
            with accelerator.autocast():
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    logger.info(f"Dataset {dataset_name} dataloader exhausted at iteration {cur_iter}, resetting...")
                    dataloader_iters[dataset_name] = iter(dataloader_dict[dataset_name])
                    batch = next(dataloader_iters[dataset_name])
                    
                images, annotations, metas = batch.values()
                if dataset_name in ["VideoCaption"]:
                    text = [annotation['text'] for annotation in annotations]
                else:
                    text = None
                
                if epoch < lr_warmup_epochs:
                    # Do warmup:
                    lr_warmup_multi_groups(
                        optimizer=optimizer,
                        epoch=epoch, curr_iter=cur_iter,
                        warmup_epochs=lr_warmup_epochs, num_iter_per_epoch=max_iterations,
                    )
                
                outputs = model(images, dataset_name, metas, text)
                
                loss_outputs = {}
                for head in datasets_to_heads[dataset_name]:
                    if head == "VideoCaption":
                        loss_outputs[head] = loss_fn_dict[head](outputs[head], annotations, metas)
                    else:
                        loss_outputs[head] = loss_fn_dict[head](outputs[head], annotations)
                    
                for head_name in datasets_to_heads[dataset_name]:
                    loss_output = loss_outputs[head_name]
                    if head_name in ["SoccerNetGSR_Detection"]:
                        loss_task_raw, weight_dict, _ = loss_output
                    elif head_name in ["VideoCaption", "CaptionClassification", "KeypointsDetection", "LinesDetection"]:
                        loss_task_raw, weight_dict = loss_output
                    else:
                        raise ValueError(f"Head {head_name} not supported.")
                    unweighted_loss_dict[head_name] = {k: v for k, v in loss_task_raw.items() if k in weight_dict}
                    weighted_loss_dict[head_name] = {k: (v * weight_dict[k]) for k, v in loss_task_raw.items() if k in weight_dict}
                    log_only_loss_dict[head_name] = {k: v for k, v in loss_task_raw.items() if k not in weight_dict}
                    
            dataset_total_loss = sum(sum(weighted_loss_dict[head].values()) for head in datasets_to_heads[dataset_name])
            accelerator.backward(dataset_total_loss)

        if (cur_iter + 1) % accumulate_steps == 0:
            if use_clip_grad_norm:
                if use_accelerate_clip_norm:
                    original_model = model.module if hasattr(model, 'module') else model
                    backbone_grad_norm = accelerator.clip_grad_norm_(original_model.backbone.parameters(), max_norm=max_clip_norm)
                    head_grad_norms = {}
                    for head_name, head in original_model.multi_task_head.items():
                        head_grad_norms[head_name] = accelerator.clip_grad_norm_(head.parameters(), max_norm=max_clip_norm)
                else:
                    accelerator.unscale_gradients()
                    original_model = model.module if hasattr(model, 'module') else model
                    backbone_grad_norm = torch.nn.utils.clip_grad_norm_(original_model.backbone.parameters(), max_clip_norm)
                    head_grad_norms = {}
                    for head_name, head in original_model.multi_task_head.items():
                        head_grad_norms[head_name] = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=max_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
                
        states["global_step"] += 1
        tps.update(TPS.timestamp() - step_timestamp)
        step_timestamp = TPS.timestamp()
        
        if logger and (cur_iter + 1) % logging_interval == 0:
            total_weighted_loss = sum(sum(task_losses.values()) for task_losses in weighted_loss_dict.values())
            total_unweighted_loss = sum(sum(task_losses.values()) for task_losses in unweighted_loss_dict.values())
            logger.log_scalar("train_overall/weighted_total_loss", total_weighted_loss, states["global_step"])
            logger.log_scalar("train_overall/unweighted_total_loss", total_unweighted_loss, states["global_step"])
            
            for head_name in weighted_loss_dict.keys():
                if weighted_loss_dict[head_name]:
                    logger.log_loss_dict(weighted_loss_dict[head_name], states["global_step"], prefix=f"train_weighted_{head_name}")
                if unweighted_loss_dict[head_name]:
                    logger.log_loss_dict(unweighted_loss_dict[head_name], states["global_step"], prefix=f"train_unweighted_{head_name}")
                if log_only_loss_dict[head_name]:
                    logger.log_loss_dict(log_only_loss_dict[head_name], states["global_step"], prefix=f"train_unweighted_{head_name}", count_sum=False)
            
            logger.log_learning_rate(optimizer, states["global_step"])
            if use_clip_grad_norm:
                logger.log_scalar("train_grad_norm/backbone_grad_norm", backbone_grad_norm, states["global_step"])
                for head_name, head_grad_norm in head_grad_norms.items():
                    logger.log_scalar(f"train_grad_norm/{head_name}_head_grad_norm", head_grad_norm, states["global_step"])
            
            for head_name, task_losses in weighted_loss_dict.items():
                for loss_name, loss_value in task_losses.items():
                    epoch_metrics.update({f"weighted_{head_name}_{loss_name}": loss_value})
            
            for head_name, task_losses in unweighted_loss_dict.items():
                for loss_name, loss_value in task_losses.items():
                    epoch_metrics.update({f"unweighted_{head_name}_{loss_name}": loss_value})
            
            for head_name, task_losses in log_only_loss_dict.items():
                for loss_name, loss_value in task_losses.items():
                    epoch_metrics.update({f"{head_name}_{loss_name}": loss_value})
            
            metrics.update(name="weighted_total_loss", value=total_weighted_loss.detach())
            metrics.update(name="unweighted_total_loss", value=total_unweighted_loss.detach())
            
            for head_name, task_losses in weighted_loss_dict.items():
                for loss_name, loss_value in task_losses.items():
                    metrics.update(name=f"weighted_{head_name}_{loss_name}", value=loss_value.detach())
            
            for head_name, task_losses in unweighted_loss_dict.items():
                for loss_name, loss_value in task_losses.items():
                    metrics.update(name=f"unweighted_{head_name}_{loss_name}", value=loss_value.detach())
            
            for head_name, task_losses in log_only_loss_dict.items():
                for loss_name, loss_value in task_losses.items():
                    metrics.update(name=f"{head_name}_{loss_name}", value=loss_value.detach())
            
            for i, param_group in enumerate(optimizer.param_groups):
                group_name = param_group['name']
                _lr = param_group['lr']
                metrics[f"lr_{group_name}"].clear()
                metrics.update(name=f"lr_{group_name}", value=_lr)
            if use_clip_grad_norm:
                metrics.update(name="backbone_grad_norm", value=backbone_grad_norm.detach())
                for head_name, head_grad_norm in head_grad_norms.items():
                    metrics.update(name=f"{head_name}_head_grad_norm", value=head_grad_norm.detach())
            torch.cuda.synchronize()
            _cuda_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            _cuda_memory = torch.tensor([_cuda_memory], device=device)
            _gathered_cuda_memory = accelerator.gather(_cuda_memory)
            _max_cuda_memory = _gathered_cuda_memory.max().item()
            accelerator.wait_for_everyone()
            metrics["max_cuda_mem(MB)"].clear()
            metrics.update(name="max_cuda_mem(MB)", value=_max_cuda_memory)
            metrics.sync()
            eta = tps.eta(total_steps=max_iterations, current_steps=cur_iter)
            eta = TPS.format(eta)
            logger.metrics(
                log=f"[Epoch: {epoch}] [{cur_iter}/{max_iterations}] "
                    f"[tps: {tps.average:.2f}s] [eta: {eta}] ",
                metrics=metrics,
                global_step=states["global_step"],
            )
        
    states["start_epoch"] += 1
    time_per_epoch = TPS.format(TPS.timestamp() - epoch_start_timestamp)
    if logger:
        epoch_avg_metrics = epoch_metrics.get_averages()
        epoch_weighted_total = 0.0
        epoch_unweighted_total = 0.0
        
        for key, value in epoch_avg_metrics.items():
            if key.startswith("weighted_") and not key.startswith("weighted_total"):
                epoch_weighted_total += value
            elif key.startswith("unweighted_") and not key.startswith("unweighted_total"):
                epoch_unweighted_total += value
        
        logger.log_scalar("epoch_overall/weighted_total_loss", epoch_weighted_total, epoch)
        logger.log_scalar("epoch_overall/unweighted_total_loss", epoch_unweighted_total, epoch)
        
        task_weighted_totals = {}
        task_unweighted_totals = {}
        
        for key, value in epoch_avg_metrics.items():
            if key.startswith("weighted_"):
                parts = key.split("_", 2)
                if len(parts) >= 3:
                    head_name = parts[1]
                    if head_name not in task_weighted_totals:
                        task_weighted_totals[head_name] = 0.0
                    task_weighted_totals[head_name] += value
            elif key.startswith("unweighted_"):
                parts = key.split("_", 2)
                if len(parts) >= 3:
                    head_name = parts[1]
                    if head_name not in task_unweighted_totals:
                        task_unweighted_totals[head_name] = 0.0
                    task_unweighted_totals[head_name] += value
        
        for head_name, total_loss in task_weighted_totals.items():
            logger.log_scalar(f"epoch_weighted_{head_name}/total_loss", total_loss, epoch)
        
        for head_name, total_loss in task_unweighted_totals.items():
            logger.log_scalar(f"epoch_unweighted_{head_name}/total_loss", total_loss, epoch)
        
        logger.flush_tb_writer()
        logger.info(f"Epoch {epoch} completed. Time per epoch: {time_per_epoch}")

def lr_warmup_multi_groups(optimizer, epoch: int, curr_iter: int, warmup_epochs: int, num_iter_per_epoch: int):
    """
    Learning rate warmup for multiple parameter groups with different target learning rates.
    Each parameter group's initial learning rate is used as the target learning rate.
    """
    total_warmup_iters = warmup_epochs * num_iter_per_epoch
    current_lr_ratio = (epoch * num_iter_per_epoch + curr_iter + 1) / total_warmup_iters
    
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']
        
        tgt_lr = param_group['initial_lr']
        current_lr = tgt_lr * current_lr_ratio
        
        if "lr_scale" in param_group:
            param_group["lr"] = current_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = current_lr
    return

def lr_warmup(optimizer, epoch: int, curr_iter: int, tgt_lr: float, warmup_epochs: int, num_iter_per_epoch: int):
    total_warmup_iters = warmup_epochs * num_iter_per_epoch
    current_lr_ratio = (epoch * num_iter_per_epoch + curr_iter + 1) / total_warmup_iters
    current_lr = tgt_lr * current_lr_ratio
    for param_grop in optimizer.param_groups:
        if "lr_scale" in param_grop:
            param_grop["lr"] = current_lr * param_grop["lr_scale"]
        else:
            param_grop["lr"] = current_lr
        pass
    return
    
if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)

    if opt.super_config_path is not None:
        cfg = load_super_config(cfg, opt.super_config_path)
    else:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    cfg = update_config(config=cfg, option=opt)

    train_engine(config=cfg)