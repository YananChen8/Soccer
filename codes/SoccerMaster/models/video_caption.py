from __future__ import division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
from models.utils.flatten_data import flatten_data
from accelerate.utils.operations import gather_object

class VideoCaptionHead(nn.Module):
    def __init__(self, loss_type='siglip_loss', backbone_type='image'):
        super().__init__()
        self.loss_type = loss_type
        self.backbone_type = backbone_type
        
        if loss_type == 'siglip_loss':
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))
            self.logits_bias = nn.Parameter(torch.tensor(-10.0))
        elif loss_type == 'infonce_loss':
            self.temperature = nn.Parameter(torch.tensor(0.3))
        else:
            raise ValueError(f"Loss type {loss_type} not supported.")

    def forward(self, 
                backbone_outputs,
                metas,
                gather_distributed: bool = True):
        global_features = backbone_outputs['global_features']
        text_features = backbone_outputs['text_features']
        
        if self.backbone_type == 'video':
            vision_features = global_features.mean(dim=1)  # [N, D]
        else:
            vision_features = global_features[:, 0]  # [N, D]
        
        vision_features = F.normalize(vision_features, dim=-1)
        
        assert text_features is not None
        # Detect valid text features (non-zero vectors)
        text_norms = torch.norm(text_features, dim=-1)
        valid_text_mask = text_norms > 1e-6
        
        # Always gather regardless of validity to avoid DDP deadlock
        text_features = F.normalize(text_features, dim=-1)
        
        if dist.is_initialized() and gather_distributed:
            vision_features, text_features, valid_text_mask = self._gather_features_distributed_with_mask(
                vision_features, 
                text_features,
                valid_text_mask
            )
        
        if valid_text_mask.any():
            base_similarity_matrix = vision_features @ text_features.t()
            
            if self.loss_type == 'siglip_loss':
                processed_similarity_matrix = base_similarity_matrix * self.logit_scale.exp() + self.logits_bias
            elif self.loss_type == 'infonce_loss':
                processed_similarity_matrix = base_similarity_matrix / self.temperature.clamp(min=1e-6)
            else:
                processed_similarity_matrix = base_similarity_matrix
        else:
            base_similarity_matrix = None
            processed_similarity_matrix = None
            valid_text_mask = torch.zeros(vision_features.shape[0], dtype=torch.bool, device=vision_features.device)
        
        output = {
            'vision_features': vision_features,
            'text_features': text_features,
            'base_similarity_matrix': base_similarity_matrix,
            'processed_similarity_matrix': processed_similarity_matrix,
            'valid_text_mask': valid_text_mask
        }
        return output
    
    def _gather_features_distributed(self, vision, text):
        """Gather feature tensors across GPUs. Returns shape: [world_size * local_batch, D]"""
        world_size = dist.get_world_size()
        vision_list = [torch.zeros_like(vision) for _ in range(world_size)]
        text_list = [torch.zeros_like(text) for _ in range(world_size)]
        dist.all_gather(vision_list, vision)
        dist.all_gather(text_list, text)
        # Preserve gradients for current process features
        vision_list[dist.get_rank()] = vision
        text_list[dist.get_rank()] = text
        return torch.cat(vision_list, dim=0), torch.cat(text_list, dim=0)
    
    def _gather_features_distributed_with_mask(self, vision, text, valid_mask):
        """Gather feature tensors and mask across GPUs. Returns shape: [world_size * local_batch, D]"""
        world_size = dist.get_world_size()
        vision_list = [torch.zeros_like(vision) for _ in range(world_size)]
        text_list = [torch.zeros_like(text) for _ in range(world_size)]
        mask_list = [torch.zeros_like(valid_mask) for _ in range(world_size)]
        dist.all_gather(vision_list, vision)
        dist.all_gather(text_list, text)
        dist.all_gather(mask_list, valid_mask)
        # Preserve gradients for current process features
        vision_list[dist.get_rank()] = vision
        text_list[dist.get_rank()] = text
        mask_list[dist.get_rank()] = valid_mask
        return torch.cat(vision_list, dim=0), torch.cat(text_list, dim=0), torch.cat(mask_list, dim=0)

class VideoCaptionLoss(nn.Module):
    def __init__(self, 
                 weight_dict, 
                 loss_type='siglip_loss', 
                 distributed_gather=True):
        super().__init__()
        self.weight_dict = weight_dict
        self.loss_type = loss_type
        self.distributed_gather = distributed_gather

    def forward(self, outputs, targets, metas=None):
        valid_text_mask = outputs.get('valid_text_mask', None)
        proc_sim = outputs['processed_similarity_matrix']
        base_sim = outputs['base_similarity_matrix']
        
        if valid_text_mask is None or not valid_text_mask.any() or proc_sim is None or base_sim is None:
            device = next(iter(outputs.values())).device if outputs else torch.device('cpu')
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=device)
            losses = {
                f'{self.loss_type}': dummy_loss,
                'top_1_accuracy': torch.tensor(0.0, device=device),
                'top_3_accuracy': torch.tensor(0.0, device=device),
                'top_5_accuracy': torch.tensor(0.0, device=device),
                'top_1_accuracy_type': torch.tensor(0.0, device=device),
                'top_3_accuracy_type': torch.tensor(0.0, device=device),
                'top_5_accuracy_type': torch.tensor(0.0, device=device)
            }
            return losses, self.weight_dict
        
        global_captions = self._gather_captions_distributed(targets)
        valid_indices = torch.where(valid_text_mask)[0]
        valid_captions = [global_captions[i] for i in valid_indices.cpu().tolist()]
        valid_proc_sim = proc_sim[valid_text_mask][:, valid_text_mask]
        valid_base_sim = base_sim[valid_text_mask][:, valid_text_mask]
        
        target_label = create_label_from_comment(valid_captions).to(valid_base_sim.device)
        
        if self.loss_type == 'siglip_loss':
            loss = self.compute_siglip_loss(valid_proc_sim, target_label)
        elif self.loss_type == 'infonce_loss':
            loss = self.compute_infonce_loss(valid_proc_sim, target_label)
        else:
            raise ValueError(f"Unsupported loss: {self.loss_type}")
        
        top_1_acc, top_3_acc, top_5_acc = self.calculate_top_k_accuracy(valid_base_sim, target_label)
        with torch.no_grad():
            target_label_type = create_label_from_type(valid_captions).to(valid_base_sim.device)
            top_1_acc_type, top_3_acc_type, top_5_acc_type = self.calculate_top_k_accuracy(valid_base_sim, target_label_type)
        
        losses = {
            f'{self.loss_type}': loss,
            'top_1_accuracy': top_1_acc,
            'top_3_accuracy': top_3_acc,
            'top_5_accuracy': top_5_acc,
            'top_1_accuracy_type': top_1_acc_type,
            'top_3_accuracy_type': top_3_acc_type,
            'top_5_accuracy_type': top_5_acc_type
        }
        return losses, self.weight_dict
    
    def _gather_captions_distributed(self, targets):
        local_captions = [t['caption'] for t in targets]
        if dist.is_initialized() and self.distributed_gather:
            global_captions = self._gather_list_distributed(local_captions)
        else:
            global_captions = local_captions
        return global_captions
    
    def _create_global_label(self, targets, device):
        """Create global label matrix with DDP support."""
        local_captions = [t['caption'] for t in targets]
        if dist.is_initialized() and self.distributed_gather:
            global_captions = self._gather_list_distributed(local_captions)
        else:
            global_captions = local_captions
        return create_label_from_comment(global_captions).to(device)
    
    def _gather_list_distributed(self, local_list):
        world_size = dist.get_world_size()
        local_data = [local_list]
        all_data = [None] * world_size
        dist.all_gather_object(all_data, local_data)
        flat_list = []
        for data in all_data:
            flat_list.extend(data[0])
        return flat_list
    
    def compute_siglip_loss(self, logits, target_label):
        """SigLIP loss: compute per-sample loss and average.
        Args:
            logits: [batch_size, batch_size] similarity matrix
            target_label: [batch_size, batch_size] label matrix (1=positive, -1=negative)
        """
        logits_per_image = logits.t()
        return -F.logsigmoid(target_label * logits_per_image).sum() / logits.shape[0]

    def compute_infonce_loss(self, logits, target_label):
        logits = logits / self.temperature
        pos_mask = (target_label > 0).float()
        neg_mask = (target_label < 0).float()
        
        pos_loss = -torch.log(torch.sigmoid(logits)) * pos_mask
        neg_loss = -torch.log(1 - torch.sigmoid(logits)) * neg_mask
        
        n_pos = pos_mask.sum().clamp(min=1)
        n_neg = neg_mask.sum().clamp(min=1)
        
        return (pos_loss.sum() / n_pos + neg_loss.sum() / n_neg) / 2

    def calculate_top_k_accuracy(self, sim_matrix, labels):
        batch_size = sim_matrix.size(0)
        max_k = min(5, batch_size)
        topk_indices = torch.topk(sim_matrix, k=max_k, dim=1)[1]
        pos_mask = (labels > 0).float()
        correct = torch.gather(pos_mask, 1, topk_indices)
        
        top1_acc = correct[:, 0].sum() / batch_size
        top3_acc = (torch.tensor(1.0, device=sim_matrix.device) if batch_size < 3
                    else correct[:, :3].sum(dim=1).clamp(max=1).sum() / batch_size)
        top5_acc = (torch.tensor(1.0, device=sim_matrix.device) if batch_size < 5
                    else correct.sum(dim=1).clamp(max=1).sum() / batch_size)
        
        return top1_acc, top3_acc, top5_acc

def create_label_from_comment(captions, special_categories=None):
    """Create global label matrix: 1=positive, -1=negative.
    Special event categories that are semantically equivalent are also labeled positive.
    """
    N = len(captions)
    labels = -torch.ones(N, N)
    
    if special_categories is None:
        special_categories = {"end of half game", "off side", "start of half game",
                             "ball possession", "substitution"}
    
    for i in range(N):
        labels[i, i] = 1.0
        
    for i, cap_i in enumerate(captions):
        for j, cap_j in enumerate(captions):
            if cap_i in special_categories and cap_i == cap_j:
                labels[i, j] = 1.0
                labels[j, i] = 1.0
    
    return labels

def create_label_from_type(captions):
    """Create global label matrix: 1=positive if same caption type, -1=negative."""
    N = len(captions)
    labels = -torch.ones(N, N)
    
    for i in range(N):
        labels[i, i] = 1.0
        
    for i, cap_i in enumerate(captions):
        for j, cap_j in enumerate(captions):
            if cap_i == cap_j:
                labels[i, j] = 1.0
                labels[j, i] = 1.0
    
    return labels

def build_video_caption_loss(config):
    loss_type = config["VIDEO_CAPTION_LOSS_TYPE"]
    
    if loss_type == "siglip_loss":
        weight_dict = {'siglip_loss': config["VIDEO_CAPTION_SIGLIP_LOSS_WEIGHT"]}
    elif loss_type == "infonce_loss":
        weight_dict = {'infonce_loss': config["VIDEO_CAPTION_INFONCE_LOSS_WEIGHT"]}
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return VideoCaptionLoss(
        weight_dict=weight_dict,
        loss_type=loss_type,
        distributed_gather=config["DISTRIBUTED_GATHER"]
    )

def build_video_caption_head(config):
    return VideoCaptionHead(
        loss_type=config["VIDEO_CAPTION_LOSS_TYPE"],
        backbone_type=config["BACKBONE_TYPE"]
    )


class VideoCaptionMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.reset()
        
    def reset(self):
        self.video_caption_metrics_data = {
            'top_1_accuracy': [],
            'top_3_accuracy': [],
            'top_5_accuracy': [],
            'top_1_accuracy_type': [],
            'top_3_accuracy_type': [],
            'top_5_accuracy_type': [],
            'retrieval_batch_size': [],
            'sample_count': 0
        }

    def update(self, outputs, targets, loss_task_raw=None):
        valid_text_mask = outputs.get('valid_text_mask', None)
        if valid_text_mask is None or not valid_text_mask.any():
            return
        
        top_1_acc = loss_task_raw['top_1_accuracy']
        top_3_acc = loss_task_raw['top_3_accuracy'] 
        top_5_acc = loss_task_raw['top_5_accuracy']
        top_1_acc_type = loss_task_raw['top_1_accuracy_type']
        top_3_acc_type = loss_task_raw['top_3_accuracy_type']
        top_5_acc_type = loss_task_raw['top_5_accuracy_type']
        
        self.video_caption_metrics_data['top_1_accuracy'].append(top_1_acc.cpu().item())
        self.video_caption_metrics_data['top_3_accuracy'].append(top_3_acc.cpu().item())
        self.video_caption_metrics_data['top_5_accuracy'].append(top_5_acc.cpu().item())
        self.video_caption_metrics_data['top_1_accuracy_type'].append(top_1_acc_type.cpu().item())
        self.video_caption_metrics_data['top_3_accuracy_type'].append(top_3_acc_type.cpu().item())
        self.video_caption_metrics_data['top_5_accuracy_type'].append(top_5_acc_type.cpu().item())
        
        valid_sample_count = valid_text_mask.sum().item()
        self.video_caption_metrics_data['sample_count'] += valid_sample_count
        self.video_caption_metrics_data['retrieval_batch_size'].append(valid_sample_count)

    def gather_metrics_data(self, accelerator):
        video_caption_key_list = ['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy', 'top_1_accuracy_type', 'top_3_accuracy_type', 'top_5_accuracy_type', 'retrieval_batch_size']
        gathered_video_caption_metrics = {}
        
        for key in video_caption_key_list:
            gathered_video_caption_metrics[key] = gather_object(self.video_caption_metrics_data[key])
        
        gathered_video_caption_metrics['sample_count'] = gather_object([self.video_caption_metrics_data['sample_count']])
        return gathered_video_caption_metrics

    def compute_metrics_from_gathered_data(self, gathered_video_caption_metrics):
        metrics = {}
        all_top_1_accuracy = flatten_data(gathered_video_caption_metrics['top_1_accuracy'])
        all_top_3_accuracy = flatten_data(gathered_video_caption_metrics['top_3_accuracy'])
        all_top_5_accuracy = flatten_data(gathered_video_caption_metrics['top_5_accuracy'])
        all_top_1_accuracy_type = flatten_data(gathered_video_caption_metrics['top_1_accuracy_type'])
        all_top_3_accuracy_type = flatten_data(gathered_video_caption_metrics['top_3_accuracy_type'])
        all_top_5_accuracy_type = flatten_data(gathered_video_caption_metrics['top_5_accuracy_type'])
        all_retrieval_batch_size = flatten_data(gathered_video_caption_metrics['retrieval_batch_size'])
        
        total_sample_count = sum(gathered_video_caption_metrics['sample_count'])
        
        if total_sample_count > 0:
            top_1_accuracy_tensor = torch.tensor(all_top_1_accuracy, dtype=torch.float32)
            top_3_accuracy_tensor = torch.tensor(all_top_3_accuracy, dtype=torch.float32)
            top_5_accuracy_tensor = torch.tensor(all_top_5_accuracy, dtype=torch.float32)
            top_1_accuracy_type_tensor = torch.tensor(all_top_1_accuracy_type, dtype=torch.float32)
            top_3_accuracy_type_tensor = torch.tensor(all_top_3_accuracy_type, dtype=torch.float32)
            top_5_accuracy_type_tensor = torch.tensor(all_top_5_accuracy_type, dtype=torch.float32)
            retrieval_batch_size_tensor = torch.tensor(all_retrieval_batch_size, dtype=torch.float32)
            
            metrics['video_caption_top_1_accuracy'] = top_1_accuracy_tensor.mean().item()
            metrics['video_caption_top_3_accuracy'] = top_3_accuracy_tensor.mean().item()
            metrics['video_caption_top_5_accuracy'] = top_5_accuracy_tensor.mean().item()
            metrics['video_caption_top_1_accuracy_type'] = top_1_accuracy_type_tensor.mean().item()
            metrics['video_caption_top_3_accuracy_type'] = top_3_accuracy_type_tensor.mean().item()
            metrics['video_caption_top_5_accuracy_type'] = top_5_accuracy_type_tensor.mean().item()
            metrics['video_caption_retrieval_batch_size'] = retrieval_batch_size_tensor.mean().item()
            
            metrics['video_caption_total_samples'] = total_sample_count
            metrics['video_caption_top_1_accuracy_std'] = top_1_accuracy_tensor.std().item()
            metrics['video_caption_top_3_accuracy_std'] = top_3_accuracy_tensor.std().item()
            metrics['video_caption_top_5_accuracy_std'] = top_5_accuracy_tensor.std().item()
            metrics['video_caption_top_1_accuracy_type_std'] = top_1_accuracy_type_tensor.std().item()
            metrics['video_caption_top_3_accuracy_type_std'] = top_3_accuracy_type_tensor.std().item()
            metrics['video_caption_top_5_accuracy_type_std'] = top_5_accuracy_type_tensor.std().item()
            
        else:
            for metric_name in ['video_caption_top_1_accuracy', 'video_caption_top_3_accuracy', 'video_caption_top_5_accuracy',
                                'video_caption_top_1_accuracy_type', 'video_caption_top_3_accuracy_type', 'video_caption_top_5_accuracy_type', 'video_caption_retrieval_batch_size',
                                'video_caption_top_1_accuracy_std', 'video_caption_top_3_accuracy_std', 'video_caption_top_5_accuracy_std',
                                'video_caption_top_1_accuracy_type_std', 'video_caption_top_3_accuracy_type_std', 'video_caption_top_5_accuracy_type_std']:
                metrics[metric_name] = 0.0
            metrics['video_caption_total_samples'] = 0
        
        return metrics

    @torch.no_grad()
    def forward(self, outputs, targets, loss_task_raw=None):
        self.update(outputs, targets, loss_task_raw)
        return {}

    def compute_final_metrics(self, accelerator):
        gathered_data = self.gather_metrics_data(accelerator)
        if accelerator.is_main_process:
            return self.compute_metrics_from_gathered_data(gathered_data)
        return {}


def build_video_caption_metrics(config):
    return VideoCaptionMetrics()