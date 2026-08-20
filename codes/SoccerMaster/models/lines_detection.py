# ------------------------------------------------------------------------
# Modified from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import os
from typing import List, Tuple
import copy

from models.utils.flatten_data import flatten_data
from accelerate.utils.operations import gather_object

class LinesDetection(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone_num_channels, num_lines, backbone_type='image', head_type='default', selected_layers=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            head_type: str, 'default' for LinesHead
            selected_layers: list
        """
        # TODO: find a way to handle positional encoding, strides, channels, etc.
        super().__init__()
        self.backbone_type = backbone_type
        self.head_type = head_type
        self.selected_layers = selected_layers
        self.lines_head = LinesHead(dim_in=backbone_num_channels[0], num_lines=num_lines)

    def forward(self, backbone_outputs, metas, is_training: bool = False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        global_features = backbone_outputs['global_features']
        local_features = backbone_outputs['local_features']
        
        bs, num_frames = None, None
        if self.backbone_type == 'video':
            bs, num_frames, _, _ = local_features.shape
            local_features = local_features.reshape(bs * num_frames, *local_features.shape[2:])
            if global_features is not None:
                global_features = global_features.reshape(bs * num_frames, -1)

        N, L, D = local_features.shape
        reshaped_local_features = local_features.permute(0, 2, 1).contiguous()
        Hf = Wf = int(math.sqrt(L))
        reshaped_local_features = reshaped_local_features.reshape(N, D, Hf, Wf)
        
        lines_heatmap = self.lines_head(reshaped_local_features)

        if self.backbone_type == 'video':
            lines_heatmap = lines_heatmap.reshape(bs, num_frames, *lines_heatmap.shape[1:])

        out = {'pred_lines_heatmap': lines_heatmap}
        return out

class LinesDetectionLoss(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, weight_dict, backbone_type='image'):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            backbone_type: 'image' or 'video'
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.backbone_type = backbone_type

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
                      For video mode: list of lists, where each inner list contains annotations for each frame
        """
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
        
        valid_lines_mask = torch.stack([t["valid_lines"] for t in targets_for_loss], dim=0)
        
        if valid_lines_mask.any():
            lines_gt = torch.stack([t["lines_target"] for t in targets_for_loss], dim=0)
            lines_pred = outputs["pred_lines_heatmap"]
            
            if self.backbone_type == 'video' and len(lines_pred.shape) == 5:
                bs, num_frames = lines_pred.shape[:2]
                lines_pred = lines_pred.reshape(bs * num_frames, *lines_pred.shape[2:])
            
            expanded_mask = valid_lines_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(lines_gt)
            loss_lines = F.mse_loss(lines_pred * expanded_mask, lines_gt * expanded_mask, reduction='sum')
            valid_elements = expanded_mask.sum()
            if valid_elements > 0:
                loss_lines = loss_lines / valid_elements
            else:
                loss_lines = torch.tensor(0.0, device=lines_pred.device, requires_grad=True)
        else:
            loss_lines = torch.tensor(0.0, device=outputs["pred_lines_heatmap"].device, requires_grad=True)
        
        losses["loss_lines"] = loss_lines
        return losses, self.weight_dict

class LinesHead(nn.Module):
    def __init__(self, dim_in=768, num_lines=24):
        super(LinesHead, self).__init__()
        self.dim_in = dim_in
        # Using sub-pixel convolution (pixel shuffle) for learnable upsampling
        # This is more parameter-efficient and often works better than transposed convolution
        
        # Stage 1: (768, 32, 32) -> (192, 64, 64) using 2x upsampling
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
            nn.Conv2d(48, num_lines, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_lines),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_lines, num_lines, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
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
        Forward pass using learnable upsampling
        Args:
            x: Input features of shape (N, 768, 32, 32)
        Returns:
            output: Reconstructed features of shape (N, output_channels, 512, 512)
        """
        x = self.stage1(x)      # (N, 192, 64, 64)
        x = self.stage2(x)      # (N, 96, 128, 128)
        x = self.stage3(x)      # (N, 48, 256, 256)
        x = self.final_conv(x)
        
        return x


class LinesDetectionMetrics(nn.Module):
    """Computes lines metrics including accuracy, precision, recall and F1, with multi-process aggregation support."""
    def __init__(self, backbone_type='image'):
        super().__init__()
        self.backbone_type = backbone_type
        
        self.lines_metrics_data = {
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'f1_scores': [],
            'valid_count': 0
        }
        
    def reset(self):
        self.lines_metrics_data = {
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'f1_scores': [],
            'valid_count': 0
        }

    def get_keypoints_from_heatmap_batch_maxpool_l(
            self,
            heatmap: torch.Tensor,
            scale: int = 2,
            max_keypoints: int = 2,
            min_keypoint_pixel_distance: int = 10,
            return_scores: bool = True,
    ) -> List[List[List[Tuple[int, int]]]]:
        """Fast extraction of keypoints from a batch of heatmaps using maxpooling.

        Inspired by mmdetection and CenterNet:
        https://mmdetection.readthedocs.io/en/v2.13.0/_modules/mmdet/models/utils/gaussian_target.html

        Args:
            heatmap (torch.Tensor): NxCxHxW heatmap batch
            max_keypoints (int, optional): max number of keypoints to extract, lowering will result in faster execution times. Defaults to 20.
            min_keypoint_pixel_distance (int, optional): _description_. Defaults to 1.

            Following thresholds can be used at inference time to select where you want to be on the AP curve. They should ofc. not be used for training
            abs_max_threshold (Optional[float], optional): _description_. Defaults to None.
            rel_max_threshold (Optional[float], optional): _description_. Defaults to None.

        Returns:
            The extracted keypoints for each batch, channel and heatmap; and their scores
        """
        batch_size, n_channels, _, width = heatmap.shape
        kernel = min_keypoint_pixel_distance * 2 + 1
        pad = int((kernel-1)/2)

        max_pooled_heatmap = torch.nn.functional.max_pool2d(heatmap, kernel, stride=1, padding=pad)
        # if the value equals the original value, it is the local maximum
        local_maxima = max_pooled_heatmap == heatmap

        # all values to zero that are not local maxima
        heatmap = heatmap * local_maxima

        # extract top-k from heatmap (may include non-local maxima if there are less peaks than max_keypoints)
        scores, indices = torch.topk(heatmap.view(batch_size, n_channels, -1), max_keypoints, sorted=True)
        indices = torch.stack([torch.div(indices, width, rounding_mode="floor"), indices % width], dim=-1)
        # at this point either score > 0.0, in which case the index is a local maximum
        # or score is 0.0, in which case topk returned non-maxima, which will be filtered out later.

        #  remove top-k that are not local maxima and threshold (if required)
        # thresholding shouldn't be done during training

        #  moving them to CPU now to avoid multiple GPU-mem accesses!
        indices = indices.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        filtered_indices = [[[] for _ in range(n_channels)] for _ in range(batch_size)]
        filtered_scores = [[[] for _ in range(n_channels)] for _ in range(batch_size)]

        # have to do this manually as the number of maxima for each channel can be different
        for batch_idx in range(batch_size):
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
                filtered_indices[batch_idx][channel_idx] = locs

        return torch.tensor(filtered_indices)

    def calculate_lines_metrics(self, gt, pred, conf_th=0.1, dist_th=5):
        gt = gt.cpu()
        pred = pred.cpu()
        
        batch_size = gt.shape[0]
        batch_metrics = []
        
        for batch_idx in range(batch_size):
            # Get data for current batch
            gt_batch = gt[batch_idx]  # [num_lines, max_keypoints, 3]
            pred_batch = pred[batch_idx]  # [num_lines, max_keypoints, 3]
            
            # Extract positions and confidence scores
            pred_pos = pred_batch[:, :, :-1]  # [num_lines, max_keypoints, 2]
            gt_pos = gt_batch[:, :, :-1]  # [num_lines, max_keypoints, 2]
            
            pred_mask = torch.all((pred_batch[:, :, -1] > conf_th), dim=-1)  # [num_lines]
            gt_mask = torch.all((gt_batch[:, :, -1] > conf_th), dim=-1)  # [num_lines]
            
            gt_flip = torch.flip(gt_pos, dims=[1])  # [num_lines, max_keypoints, 2]
            
            distances1 = torch.norm(pred_pos - gt_pos, dim=-1)  # [num_lines, max_keypoints]
            distances2 = torch.norm(pred_pos - gt_flip, dim=-1)  # [num_lines, max_keypoints]
            
            distances1_bool = torch.all((distances1 < dist_th), dim=-1)  # [num_lines]
            distances2_bool = torch.all((distances2 < dist_th), dim=-1)  # [num_lines]
            
            # Count true positives, false positives, and false negatives based on distance threshold
            true_positives = ((distances1_bool | distances2_bool) & pred_mask & gt_mask).sum().item()
            true_negatives = (~pred_mask & ~gt_mask).sum().item()
            false_positives = (
                    (pred_mask & ~gt_mask) | ((~distances1_bool & ~distances2_bool) & pred_mask & gt_mask)).sum().item()
            false_negatives = (~pred_mask & gt_mask).sum().item()
            
            # Calculate metrics for this batch
            total_lines = gt_batch.shape[0]
            if total_lines > 0:
                accuracy = (true_positives + true_negatives) / total_lines
                precision = true_positives / (true_positives + false_positives + 1e-10)
                recall = true_positives / (true_positives + false_negatives + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            else:
                accuracy = precision = recall = f1 = 0.0
            
            batch_metrics.append((accuracy, precision, recall, f1))
        
        return batch_metrics

    def compute_lines_metrics(self, pred_lines_heatmap, targets):
        if pred_lines_heatmap is None:
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
                
            # Reshape pred_lines_heatmap from [bs, num_frames, ...] to [bs*num_frames, ...]
            if len(pred_lines_heatmap.shape) == 5:  # [bs, num_frames, num_lines, H, W]
                bs, num_frames = pred_lines_heatmap.shape[:2]
                pred_lines_heatmap = pred_lines_heatmap.reshape(bs * num_frames, *pred_lines_heatmap.shape[2:])
        else:
            targets_for_metrics = targets
        
        valid_lines_mask = torch.stack([t["valid_lines"] for t in targets_for_metrics], dim=0)
        if not valid_lines_mask.any():
            return
            
        lines_gt_list = [t["lines_target"] for i, t in enumerate(targets_for_metrics) if valid_lines_mask[i]]
        if not lines_gt_list:
            return
            
        lines_gt = torch.stack(lines_gt_list, dim=0)
        pred_lines_heatmap_valid = pred_lines_heatmap[valid_lines_mask]
        
        l_gt = self.get_keypoints_from_heatmap_batch_maxpool_l(lines_gt[:,:-1,:,:], return_scores=True, max_keypoints=2)
        lines_pred = self.get_keypoints_from_heatmap_batch_maxpool_l(pred_lines_heatmap_valid[:,:-1,:,:], return_scores=True, max_keypoints=2)
        
        batch_metrics = self.calculate_lines_metrics(l_gt, lines_pred)
        
        for accuracy, precision, recall, f1 in batch_metrics:
            self.lines_metrics_data['accuracies'].append(accuracy)
            self.lines_metrics_data['precisions'].append(precision)
            self.lines_metrics_data['recalls'].append(recall)
            self.lines_metrics_data['f1_scores'].append(f1)
        
        self.lines_metrics_data['valid_count'] += len(batch_metrics)

    def update(self, outputs, targets):
        self.compute_lines_metrics(outputs['pred_lines_heatmap'], targets)

    def gather_metrics_data(self, accelerator):
        lines_key_list = ['accuracies', 'precisions', 'recalls', 'f1_scores']
        gathered_lines_metrics = {}
        for key in lines_key_list:
            gathered_lines_metrics[key] = gather_object(self.lines_metrics_data[key])
        gathered_lines_metrics['valid_count'] = gather_object([self.lines_metrics_data['valid_count']])
        
        return gathered_lines_metrics

    def compute_metrics_from_gathered_data(self, gathered_lines_metrics):
        metrics = {}
        
        if gathered_lines_metrics is not None:
            all_accuracies = flatten_data(gathered_lines_metrics['accuracies'])
            all_precisions = flatten_data(gathered_lines_metrics['precisions'])
            all_recalls = flatten_data(gathered_lines_metrics['recalls'])
            all_f1_scores = flatten_data(gathered_lines_metrics['f1_scores'])
            total_valid_count = sum(gathered_lines_metrics['valid_count'])
            
            if total_valid_count > 0 and len(all_accuracies) > 0:
                accuracies = torch.tensor(all_accuracies, dtype=torch.float32)
                precisions = torch.tensor(all_precisions, dtype=torch.float32)
                recalls = torch.tensor(all_recalls, dtype=torch.float32)
                f1_scores = torch.tensor(all_f1_scores, dtype=torch.float32)
                
                metrics['lines_accuracy'] = accuracies.mean().item()
                metrics['lines_precision'] = precisions.mean().item()
                metrics['lines_recall'] = recalls.mean().item()
                metrics['lines_f1'] = f1_scores.mean().item()
                metrics['lines_high_accuracy_ratio'] = (accuracies > 0.8).float().mean().item()
                metrics['lines_high_f1_ratio'] = (f1_scores > 0.7).float().mean().item()
                metrics['lines_valid_samples'] = total_valid_count
            else:
                for metric_name in ['lines_accuracy', 'lines_precision', 'lines_recall', 'lines_f1',
                                  'lines_high_accuracy_ratio', 'lines_high_f1_ratio']:
                    metrics[metric_name] = 0.0
                metrics['lines_valid_samples'] = 0
        
        return metrics

    @torch.no_grad()
    def forward(self, outputs, targets):
        self.update(outputs, targets)
        return {}
        
    def compute_final_metrics(self, accelerator):
        gathered_lines_metrics = self.gather_metrics_data(accelerator)
        if accelerator.is_main_process:
            return self.compute_metrics_from_gathered_data(gathered_lines_metrics)
        return {}


def build_lines_detection_head(config: dict):
    backbone_num_channels = [config["BACKBONE_HIDDEN_DIM"]]
    num_lines = config["NUM_LINES"]
    backbone_type = config["BACKBONE_TYPE"]
    head_type = config["LINES_HEAD_TYPE"]
    selected_layers = config["DPT_SELECTED_LAYERS"]
    
    head = LinesDetection(
        backbone_num_channels=backbone_num_channels,
        num_lines=num_lines,
        backbone_type=backbone_type,
        head_type=head_type,
        selected_layers=selected_layers,
    )
    return head

def build_lines_detection_loss(config: dict):
    weight_dict = {
        "loss_lines": config["GSR_LINES_LOSS_WEIGHT"]
    }
    
    criterion = LinesDetectionLoss(weight_dict=weight_dict, backbone_type=config["BACKBONE_TYPE"])
    return criterion

def build_lines_detection_metrics(config: dict):
    metrics = LinesDetectionMetrics(backbone_type=config["BACKBONE_TYPE"])
    return metrics