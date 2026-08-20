import torch
import torch.nn.functional as F
from torch import nn
import math
from typing import List, Tuple
import copy

from models.utils.flatten_data import flatten_data
from accelerate.utils.operations import gather_object


class KeypointsDetection(nn.Module):
    def __init__(self, backbone_num_channels, num_keypoints, backbone_type='image'):
        """
        Args:
            backbone_num_channels: Backbone output channel sizes
            num_keypoints: Number of keypoints
            backbone_type: Backbone type, 'image' or 'video'
        """
        super().__init__()
        self.backbone_type = backbone_type
        self.keypoints_head = KeypointsHead(dim_in=backbone_num_channels[0], num_keypoints=num_keypoints)

    def forward(self, backbone_outputs, metas, is_training: bool = False):
        global_features = backbone_outputs['global_features']
        local_features = backbone_outputs['local_features']
        
        bs, num_frames = None, None
        if self.backbone_type == 'video':
            bs, num_frames, _, _ = local_features.shape
            local_features = local_features.reshape(bs * num_frames, *local_features.shape[2:])
            if global_features is not None:
                global_features = global_features.reshape(bs * num_frames, -1)

        # Reshape local_features from (N, L, D) to (N, D, H, W)
        N, L, D = local_features.shape
        reshaped_local_features = local_features.permute(0, 2, 1).contiguous()
        Hf = Wf = int(math.sqrt(L))
        reshaped_local_features = reshaped_local_features.reshape(N, D, Hf, Wf)
        
        keypoints_heatmap = self.keypoints_head(reshaped_local_features)

        if self.backbone_type == 'video':
            keypoints_heatmap = keypoints_heatmap.reshape(bs, num_frames, *keypoints_heatmap.shape[1:])

        out = {'pred_keypoints_heatmap': keypoints_heatmap}
        return out


class KeypointsHead(nn.Module):
    """Keypoint detection head using sub-pixel convolution for learnable upsampling"""
    
    def __init__(self, dim_in=768, num_keypoints=58):
        super(KeypointsHead, self).__init__()
        self.dim_in = dim_in
        
        # Stage 1: (768, 32, 32) -> (192, 64, 64) with 2x upsampling
        self.stage1 = nn.Sequential(
            nn.Conv2d(dim_in, 192 * 4, kernel_size=3, padding=1),  # 4x channels for 2x upsampling
            nn.PixelShuffle(2),  # (192*4, 32, 32) -> (192, 64, 64)
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: (192, 64, 64) -> (96, 128, 128)
        self.stage2 = nn.Sequential(
            nn.Conv2d(192, 96 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # (96*4, 64, 64) -> (96, 128, 128)
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3: (96, 128, 128) -> (48, 256, 256)
        self.stage3 = nn.Sequential(
            nn.Conv2d(96, 48 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # (48*4, 128, 128) -> (48, 256, 256)
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, num_keypoints, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_keypoints),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Args:
            x: Input features of shape (N, 768, 32, 32)
        Returns:
            output: Reconstructed features of shape (N, num_keypoints, 256, 256)
        """
        x = self.stage1(x)      # (N, 192, 64, 64)
        x = self.stage2(x)      # (N, 96, 128, 128)
        x = self.stage3(x)      # (N, num_keypoints, 256, 256)
        x = self.final_conv(x)  # (N, num_keypoints, 256, 256)
        
        return x


class KeypointsDetectionLoss(nn.Module):
    def __init__(self, weight_dict, backbone_type='image'):
        """
        Args:
            weight_dict: Dict containing loss weights
            backbone_type: 'image' or 'video'
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.backbone_type = backbone_type

    def forward(self, outputs, targets, **kwargs):
        losses = {}
        
        # Handle video mode: flatten targets if needed
        if self.backbone_type == 'video':
            # Check if targets is list of lists (video mode)
            if targets and isinstance(targets[0], list):
                # Flatten targets: convert list of list of dicts to list of dicts
                flattened_targets = []
                for batch_targets in targets:
                    for frame_target in batch_targets:
                        flattened_targets.append(frame_target)
                targets_for_loss = flattened_targets
            else:
                # Already flattened
                targets_for_loss = targets
        else:
            targets_for_loss = targets
        
        valid_keypoints_mask = torch.stack([t["valid_keypoints"] for t in targets_for_loss], dim=0)
        
        if valid_keypoints_mask.any():
            keypoints_gt = torch.stack([t["keypoints_target"] for t in targets_for_loss], dim=0)
            keypoints_mask = torch.stack([t["keypoints_mask"] for t in targets_for_loss], dim=0)
            keypoints_pred = outputs["pred_keypoints_heatmap"]
            
            if self.backbone_type == 'video' and len(keypoints_pred.shape) == 5:
                bs, num_frames = keypoints_pred.shape[:2]
                keypoints_pred = keypoints_pred.reshape(bs * num_frames, *keypoints_pred.shape[2:])
            
            loss_keypoints = F.mse_loss(keypoints_pred, keypoints_gt, reduction='none')
            
            keypoints_mask_expanded = keypoints_mask.unsqueeze(-1).unsqueeze(-1)
            loss_keypoints = loss_keypoints * keypoints_mask_expanded
            
            valid_sample_mask = valid_keypoints_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            valid_sample_mask = valid_sample_mask.expand_as(loss_keypoints)
            loss_keypoints = loss_keypoints * valid_sample_mask
            
            valid_keypoints_in_valid_samples = (keypoints_mask * valid_keypoints_mask.unsqueeze(1)).sum()
            if valid_keypoints_in_valid_samples > 0:
                loss_keypoints = loss_keypoints.sum() / valid_keypoints_in_valid_samples
            else:
                loss_keypoints = torch.tensor(0.0, device=keypoints_pred.device, requires_grad=True)
        else:
            loss_keypoints = torch.tensor(0.0, device=outputs["pred_keypoints_heatmap"].device, requires_grad=True)
            
        losses["loss_keypoints"] = loss_keypoints
        
        return losses, self.weight_dict


class KeypointsDetectionMetrics(nn.Module):
    def __init__(self, backbone_type='image'):
        super().__init__()
        self.backbone_type = backbone_type
        self.reset()
        
    def reset(self):
        self.keypoints_metrics_data = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'valid_count': 0
        }

    def get_keypoints_from_heatmap_batch_maxpool(
            self, 
            heatmap: torch.Tensor,
            scale: int = 2,
            max_keypoints: int = 1,
            min_keypoint_pixel_distance: int = 15,
            return_scores: bool = True,
    ):
        """Fast extraction of keypoints from a batch of heatmaps using maxpooling."""
        batch_size, n_channels, height, width = heatmap.shape

        kernel = min_keypoint_pixel_distance * 2 + 1
        pad = min_keypoint_pixel_distance
        
        # exclude border keypoints by padding with highest possible value
        padded_heatmap = torch.nn.functional.pad(heatmap, (pad, pad, pad, pad), mode="constant", value=1.0)
        max_pooled_heatmap = torch.nn.functional.max_pool2d(padded_heatmap, kernel, stride=1, padding=0)
        
        # if the value equals the original value, it is the local maximum
        local_maxima = max_pooled_heatmap == heatmap
        heatmap = heatmap * local_maxima

        # extract top-k from heatmap
        scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
        indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)

        # moving to CPU
        indices = indices.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        
        filtered_indices = []
        for batch_idx in range(batch_size):
            batch_keypoints = []
            for channel_idx in range(n_channels):
                candidates = indices[batch_idx, channel_idx]
                locs = []
                for candidate_idx in range(candidates.shape[0]):
                    # convert to (u,v)
                    loc = candidates[candidate_idx][::-1] * scale
                    loc = loc.tolist()
                    if return_scores:
                        loc.append(scores[batch_idx, channel_idx, candidate_idx])
                    locs.append(loc)
                batch_keypoints.append(locs)
            filtered_indices.append(batch_keypoints)

        return torch.tensor(filtered_indices)

    def calculate_keypoints_metrics(self, gt, pred, mask, conf_th=0.1, dist_th=5):
        # Convert mask to geometry mask (excluding last channel if needed)
        geometry_mask = (mask > 0).cpu()
            
        # Ensure gt and pred are on CPU for computation
        gt = gt.cpu()
        pred = pred.cpu()
        
        batch_size = gt.shape[0]
        batch_metrics = []
        
        for batch_idx in range(batch_size):
            if not geometry_mask[batch_idx].any():
                # No valid keypoints in this sample
                batch_metrics.append((0.0, 0.0, 0.0, 0.0))
                continue
                
            # Get valid keypoints for this batch
            valid_mask = geometry_mask[batch_idx]
            
            # Extract positions and confidence scores
            gt_batch = gt[batch_idx][valid_mask][:, 0, :]  # [valid_kp, 3]
            pred_batch = pred[batch_idx][valid_mask][:, 0, :]  # [valid_kp, 3]
            
            # Check confidence thresholds
            gt_conf_mask = gt_batch[:, -1] > conf_th  # GT confidence > threshold
            pred_conf_mask = pred_batch[:, -1] > conf_th  # Pred confidence > threshold
            
            # Calculate distances between predicted and GT positions
            gt_pos = gt_batch[:, :2]  # [valid_kp, 2] (x, y)
            pred_pos = pred_batch[:, :2]  # [valid_kp, 2] (x, y)
            distances = torch.norm(pred_pos - gt_pos, dim=1)  # [valid_kp]
            
            # Count true positives, false positives, and false negatives
            true_positives = ((distances < dist_th) & pred_conf_mask & gt_conf_mask).sum().item()
            true_negatives = (~pred_conf_mask & ~gt_conf_mask).sum().item()
            false_positives = ((pred_conf_mask & ~gt_conf_mask) | ((distances >= dist_th) & pred_conf_mask & gt_conf_mask)).sum().item()
            false_negatives = (~pred_conf_mask & gt_conf_mask).sum().item()
            
            # Calculate metrics
            total_valid = valid_mask.sum().item()
            if total_valid > 0:
                accuracy = (true_positives + true_negatives) / total_valid
                precision = true_positives / (true_positives + false_positives + 1e-10)
                recall = true_positives / (true_positives + false_negatives + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            else:
                accuracy = precision = recall = f1 = 0.0
                
            batch_metrics.append((accuracy, precision, recall, f1))
        
        return batch_metrics

    def compute_keypoints_metrics(self, pred_keypoints_heatmap, targets):
        if pred_keypoints_heatmap is None:
            return
        
        # Handle video mode: flatten targets if needed
        if self.backbone_type == 'video':
            # Check if targets is list of lists (video mode)
            if targets and isinstance(targets[0], list):
                # Flatten targets: convert list of list of dicts to list of dicts
                flattened_targets = []
                for batch_targets in targets:
                    for frame_target in batch_targets:
                        flattened_targets.append(frame_target)
                targets_for_metrics = flattened_targets
            else:
                # Already flattened
                targets_for_metrics = targets
                
            # Reshape pred_keypoints_heatmap from [bs, num_frames, ...] to [bs*num_frames, ...]
            if len(pred_keypoints_heatmap.shape) == 5:  # [bs, num_frames, num_keypoints, H, W]
                bs, num_frames = pred_keypoints_heatmap.shape[:2]
                pred_keypoints_heatmap = pred_keypoints_heatmap.reshape(bs * num_frames, *pred_keypoints_heatmap.shape[2:])
        else:
            targets_for_metrics = targets
        
        valid_keypoints_mask = torch.stack([t["valid_keypoints"] for t in targets_for_metrics], dim=0)
        if not valid_keypoints_mask.any():
            return
            
        keypoints_gt_list = [t.get("keypoints_target", None) for i, t in enumerate(targets_for_metrics) if valid_keypoints_mask[i]]
        keypoints_mask_list = [t.get("keypoints_mask", None) for i, t in enumerate(targets_for_metrics) if valid_keypoints_mask[i]]
        
        valid_data_indices = [i for i, (kp_gt, kp_mask) in enumerate(zip(keypoints_gt_list, keypoints_mask_list)) 
                        if kp_gt is not None and kp_mask is not None]
        
        if not valid_data_indices:
            return
        
        keypoints_gt = torch.stack([keypoints_gt_list[i] for i in valid_data_indices])
        keypoints_mask = torch.stack([keypoints_mask_list[i] for i in valid_data_indices])
        
        valid_pred_indices = torch.where(valid_keypoints_mask)[0][valid_data_indices]
        pred_keypoints_valid = pred_keypoints_heatmap[valid_pred_indices]
        
        kp_gt = self.get_keypoints_from_heatmap_batch_maxpool(keypoints_gt[:,:-1,:,:], return_scores=True, max_keypoints=1)
        kp_pred = self.get_keypoints_from_heatmap_batch_maxpool(pred_keypoints_valid[:,:-1,:,:], return_scores=True, max_keypoints=1)
        
        batch_metrics = self.calculate_keypoints_metrics(kp_gt, kp_pred, keypoints_mask[:, :-1])
        
        for accuracy, precision, recall, f1 in batch_metrics:
            self.keypoints_metrics_data['accuracy'].append(accuracy)
            self.keypoints_metrics_data['precision'].append(precision)
            self.keypoints_metrics_data['recall'].append(recall)
            self.keypoints_metrics_data['f1'].append(f1)
        
        self.keypoints_metrics_data['valid_count'] += len(batch_metrics)
        

    def update(self, outputs, targets):
        self.compute_keypoints_metrics(outputs['pred_keypoints_heatmap'], targets)

    def gather_metrics_data(self, accelerator):
        gathered_data = {}
        
        for key, values in self.keypoints_metrics_data.items():
            if key == 'valid_count':
                gathered_data[key] = gather_object([values])
            else:
                gathered_data[key] = gather_object(values)
        
        return gathered_data

    def compute_metrics_from_gathered_data(self, gathered_keypoints_metrics):
        keypoints_results = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            values = flatten_data(gathered_keypoints_metrics[metric_name])
            keypoints_results[f'keypoints_{metric_name}'] = sum(values) / len(values)
        keypoints_results['keypoints_valid_samples'] = sum(gathered_keypoints_metrics['valid_count'])
        return keypoints_results

    @torch.no_grad()
    def forward(self, outputs, targets):
        self.update(outputs, targets)
        return {}

    def compute_final_metrics(self, accelerator):
        gathered_keypoints_metrics = self.gather_metrics_data(accelerator)
        if accelerator.is_main_process:
            return self.compute_metrics_from_gathered_data(gathered_keypoints_metrics)
        return {}


def build_keypoints_detection_head(config: dict):
    backbone_num_channels = [config['BACKBONE_HIDDEN_DIM']]
    num_keypoints = config['NUM_KEYPOINTS']
    backbone_type = config['BACKBONE_TYPE']
    return KeypointsDetection(
        backbone_num_channels=backbone_num_channels,
        num_keypoints=num_keypoints,
        backbone_type=backbone_type,
    )


def build_keypoints_detection_loss(config: dict):
    weight_dict = {
        "loss_keypoints": config["GSR_KEYPOINTS_LOSS_WEIGHT"]
    }
    
    return KeypointsDetectionLoss(weight_dict=weight_dict, backbone_type=config["BACKBONE_TYPE"])


def build_keypoints_detection_metrics(config: dict):
    return KeypointsDetectionMetrics(backbone_type=config["BACKBONE_TYPE"])