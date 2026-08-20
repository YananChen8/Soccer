import os
import numpy as np
import torch
import cv2
import zipfile
import pickle
from sam2.build_sam import build_sam2_video_predictor
from collections import defaultdict
from tqdm import tqdm
import argparse
import json
from copy import deepcopy
import traceback
import time
import multiprocessing
from multiprocessing import Pool, Manager
from utils import (
    apply_mask_to_image,
    draw_box_with_id,
    load_tracklets_for_video, load_video_frames,
    get_next_detection,
    constrain_bbox_with_previous,
    calculate_iou,
    calculate_mask_iou,
    get_bbox_center,
    detect_outlier_bbox,
    generate_refined_pklz,
    clean_outlier_mask,
    fix_duplicate_track_ids_in_data,
    calculate_mask_connected_components,
    sam2_video_data, sam2_video_segment, sam2_detection,
    unmatched_segment, break_circular_references
)


# Environment configuration
# If using Apple MPS, fall back to CPU for unsupported operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description='SAM2 Based Inference - Multi-GPU Multi-Process')
    parser.add_argument('--sam_checkpoint', type=str, default="../checkpoints/sam2.1_hiera_large.pt", 
                        help='Path to SAM2 checkpoint')
    parser.add_argument('--sam_config', type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", 
                        help='Path to SAM2 config file')
    parser.add_argument('--input_pklz', type=str, required=True, 
                        help='Input pklz file path containing tracklets')
    parser.add_argument('--dataset_root', type=str, required=True, 
                        help='Directory containing video frames')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory for visualizations')
    parser.add_argument('--split', type=str, default='test', 
                        help='Data split (train, valid, test)')
    parser.add_argument('--video_id_list', type=str, 
                        help='Comma-separated list of specific video IDs to process (if not specified, all videos in the split will be processed), e.g., "116,117,118"')
    parser.add_argument('--video_id_start', type=int,
                        help='Starting video ID for range processing (inclusive). When used with --video_id_end, processes all video IDs from start to end.')
    parser.add_argument('--video_id_end', type=int,
                        help='Ending video ID for range processing (inclusive). When used with --video_id_start, processes all video IDs from start to end.')
    parser.add_argument('--exempt_video_id_list', type=str,
                        help='Comma-separated list of video IDs to exempt from processing (these will be excluded from output pkl and copied directly in pklz). e.g., "120,121,122"')
    parser.add_argument('--min_continuous_segment_length', type=int, default=3, 
                        help='Minimum continuous segment length to propagate')
    parser.add_argument('--min_propagate_box_dimension', type=int, default=18,
                        help='Minimum box dimension (width or height) required for propagation')
    parser.add_argument('--visualize_video', action='store_true', 
                        help='Generate visualization videos')
    parser.add_argument('--fps', type=int, default=25, 
                        help='FPS for output videos')
    parser.add_argument('--save_results', action='store_true',
                        help='Save unprocessed_detections and unmatched_segments to a pkl file')
    parser.add_argument('--output_pkl', type=str, default=None,
                        help='Path to save the output pkl file (default: <output_dir>/results.pkl)')
    parser.add_argument('--save_refined_pklz', action='store_true',
                        help='Save a refined version of the input pklz file with matched track IDs')
    parser.add_argument('--include_unmatched_segments', action='store_true',
                        help='Include unmatched segments as new rows in the refined pklz file')
    parser.add_argument('--fix_duplicate_track_ids', action='store_true',
                        help='Check and fix duplicate track_ids in each frame by adding 100')
    parser.add_argument('--save_pklz_path', type=str, default=None,
                        help='Path where the refined pklz file will be saved (default: <output_dir>/refined_<input_pklz_basename>)')
    parser.add_argument('--best_iou_threshold', type=float, default=0.5,
                        help='IoU threshold for matching segments to detections')
    parser.add_argument('--best_seg_bbox_be_overlapped_ratio_threshold', type=float, default=0.7,
                        help='Segment bbox overlap ratio threshold for matching')
    parser.add_argument('--mask_iou_threshold', type=float, default=0.6,
                        help='Mask IoU threshold for determining duplicate masks')
    parser.add_argument('--seg_mask_be_overlapped_ratio_threshold', type=float, default=0.6,
                        help='Segment mask overlap ratio threshold for determining duplicate masks')
    parser.add_argument('--max_expansion_ratio', type=float, default=1.0,
                        help='Maximum expansion ratio for bbox constraint (default: 1.0)')
    parser.add_argument('--max_width_offset', type=int, default=30,
                        help='Maximum width offset for bbox constraint (default: 30)')
    parser.add_argument('--max_height_offset', type=int, default=60,
                        help='Maximum height offset for bbox constraint (default: 60)')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='Kernel size for morphological operations in clean_outlier_mask (default: 5)')
    
    # Multi-GPU multi-process parameters
    parser.add_argument('--gpu_list', type=str, required=True,
                        help='Comma-separated list of GPU IDs to use, e.g., "0,1,2,3"')
    parser.add_argument('--max_processes_per_gpu', type=int, default=1,
                        help='Maximum number of processes per GPU (default: 1)')
    
    return parser.parse_args()

# Process-safe result collector
class ProcessSafeResultCollector:
    def __init__(self, manager):
        self.results = manager.dict()
        self.lock = manager.Lock()
        
    def add_result(self, video_id, result):
        with self.lock:
            self.results[video_id] = result
            
    def get_results(self):
        with self.lock:
            return dict(self.results)
            
    def get_result_count(self):
        with self.lock:
            return len(self.results)

# Process-safe GPU load manager
class ProcessSafeGPULoadManager:
    def __init__(self, manager, gpu_ids, max_processes_per_gpu):
        self.gpu_ids = gpu_ids
        self.max_processes_per_gpu = max_processes_per_gpu
        self.gpu_loads = manager.dict()
        for gpu_id in gpu_ids:
            self.gpu_loads[gpu_id] = 0
        self.lock = manager.Lock()
        
    def get_best_gpu(self):
        """Get the GPU ID with the lightest load"""
        with self.lock:
            available_gpus = [
                (gpu_id, self.gpu_loads[gpu_id])
                for gpu_id in self.gpu_ids
                if self.gpu_loads[gpu_id] < self.max_processes_per_gpu
            ]
            
            if not available_gpus:
                return None
            
            best_gpu, _ = min(available_gpus, key=lambda x: x[1])
            self.gpu_loads[best_gpu] += 1
            print(f"Assigned GPU {best_gpu} (current load: {self.gpu_loads[best_gpu]}/{self.max_processes_per_gpu})")
            return best_gpu
    
    def release_gpu(self, gpu_id):
        """Release GPU load"""
        with self.lock:
            if gpu_id in self.gpu_loads:
                self.gpu_loads[gpu_id] = max(0, self.gpu_loads[gpu_id] - 1)
                print(f"Released GPU {gpu_id} (current load: {self.gpu_loads[gpu_id]}/{self.max_processes_per_gpu})")
    
    def get_loads(self):
        """Get current load of all GPUs"""
        with self.lock:
            return dict(self.gpu_loads)

# Process-safe progress manager
class ProcessSafeProgressManager:
    def __init__(self, manager, total_videos):
        self.total_videos = total_videos
        self.completed_videos = manager.Value('i', 0)
        self.failed_videos = manager.list()
        self.lock = manager.Lock()
        
    def video_completed(self, video_id):
        with self.lock:
            self.completed_videos.value += 1
            print(f"✓ Video {video_id} processing completed ({self.completed_videos.value}/{self.total_videos})")
            
    def video_failed(self, video_id, error):
        with self.lock:
            self.failed_videos.append((video_id, str(error)))
            print(f"✗ Video {video_id} processing failed: {error}")
            
    def get_progress(self):
        with self.lock:
            return self.completed_videos.value, len(self.failed_videos), self.total_videos


def bbox_from_mask(mask):
    """
    Compute bbox coordinates [x_min, y_min, x_max, y_max] from a mask.
    Returns None if the mask has no valid region.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not (np.any(rows) and np.any(cols)):
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [x_min, y_min, x_max, y_max]


def process_single_frame_segment(video_segments, obj_id, frame_idx, prev_frame_idx, args):
    """
    Process the video segmentation result for a single frame.
    
    Returns:
        bool: Whether the current frame is still valid.
    """
    if frame_idx not in video_segments or obj_id not in video_segments[frame_idx]:
        return False
        
    # First constrain the current frame's bbox and mask based on the previous frame
    prev_bbox = None
    
    if (prev_frame_idx in video_segments and 
        obj_id in video_segments[prev_frame_idx] and 
        video_segments[prev_frame_idx][obj_id]['valid_mask']):
        prev_bbox = video_segments[prev_frame_idx][obj_id]['bbox']
    
    current_bbox = video_segments[frame_idx][obj_id]['bbox']
    current_mask = video_segments[frame_idx][obj_id]['mask']
    
    # Calculate the number of connected components of the current mask; skip bbox constraint when it equals 1
    num_components = calculate_mask_connected_components(current_mask)
    is_constrained = False
    
    if num_components != 1:
        constrained_mask, constrained_bbox, is_constrained = constrain_bbox_with_previous(
            current_mask, current_bbox, prev_bbox, 
            max_expansion_ratio=args.max_expansion_ratio,
            max_width_offset=args.max_width_offset,
            max_height_offset=args.max_height_offset
        )
        video_segments[frame_idx][obj_id]['mask'] = constrained_mask
        video_segments[frame_idx][obj_id]['bbox'] = constrained_bbox
        video_segments[frame_idx][obj_id]['is_constrained'] = is_constrained
    
    # Detect whether this is an outlier bbox
    is_outlier = detect_outlier_bbox(video_segments, frame_idx, obj_id)
    video_segments[frame_idx][obj_id]['has_outlier'] = is_outlier
    
    # If it is an outlier or was constrained, clean the mask and recompute the bbox
    if is_outlier or is_constrained:
        mask = video_segments[frame_idx][obj_id]['mask']
        cleaned_mask = clean_outlier_mask(mask, kernel_size=args.kernel_size)
        
        if cleaned_mask.sum() > 0:
            video_segments[frame_idx][obj_id]['mask'] = cleaned_mask
            new_bbox = bbox_from_mask(cleaned_mask)
            if new_bbox is not None:
                video_segments[frame_idx][obj_id]['bbox'] = new_bbox
            else:
                video_segments[frame_idx][obj_id]['valid_mask'] = False
                video_segments[frame_idx][obj_id]['bbox'] = None
                return False
        else:
            video_segments[frame_idx][obj_id]['valid_mask'] = False
            video_segments[frame_idx][obj_id]['bbox'] = None
            return False
    
    return True


def _invalidate_remaining_frames(video_segments, obj_id, frame_range):
    """Mark all frames within the specified range as invalid"""
    for idx in frame_range:
        if idx in video_segments and obj_id in video_segments[idx]:
            video_segments[idx][obj_id]['valid_mask'] = False
            video_segments[idx][obj_id]['bbox'] = None


def process_video_segments_in_range(video_segments, obj_id, prompt_frame_idx, valid_range_start, valid_range_end, args):
    """
    Detect and clean outlier masks within a continuous valid range, processed in two passes:
    1. From prompt_frame_idx backward to valid_range_start
    2. From prompt_frame_idx forward to valid_range_end
    """
    # First pass: from prompt_frame_idx backward to valid_range_start
    for frame_idx in range(prompt_frame_idx, valid_range_start - 1, -1):
        prev_frame_idx = frame_idx + 1 if frame_idx != prompt_frame_idx else None
        if not process_single_frame_segment(video_segments, obj_id, frame_idx, prev_frame_idx, args):
            _invalidate_remaining_frames(video_segments, obj_id, range(frame_idx - 1, valid_range_start - 1, -1))
            break
    
    # Second pass: from prompt_frame_idx forward to valid_range_end
    for frame_idx in range(prompt_frame_idx, valid_range_end + 1):
        prev_frame_idx = frame_idx - 1 if frame_idx != prompt_frame_idx else None
        if not process_single_frame_segment(video_segments, obj_id, frame_idx, prev_frame_idx, args):
            _invalidate_remaining_frames(video_segments, obj_id, range(frame_idx + 1, valid_range_end + 1))
            return False
    
    return True


def _search_valid_range_with_distance(video_segments, obj_id, prompt_frame_idx, boundary, direction, max_distance=200):
    """
    Search for the valid range boundary from prompt_frame_idx in the specified direction with distance constraint.
    
    Args:
        direction: 1 for forward search, -1 for backward search
        boundary: The boundary frame index (inclusive) for the search
    
    Returns:
        The valid range boundary frame index found
    """
    result = prompt_frame_idx
    if direction == 1:
        frame_range = range(prompt_frame_idx + 1, boundary + 1)
    else:
        frame_range = range(prompt_frame_idx - 1, boundary - 1, -1)
    
    for frame_idx in frame_range:
        if not (frame_idx in video_segments and 
                obj_id in video_segments[frame_idx] and 
                video_segments[frame_idx][obj_id]['valid_mask']):
            break
        
        current_bbox = video_segments[frame_idx][obj_id]['bbox']
        adjacent_idx = frame_idx - direction  # Adjacent frame (in the opposite direction)
        adjacent_bbox = video_segments[adjacent_idx][obj_id]['bbox']
        
        current_center = get_bbox_center(current_bbox)
        adjacent_center = get_bbox_center(adjacent_bbox)
        
        if current_center is None or adjacent_center is None:
            break
        
        distance = np.sqrt((current_center[0] - adjacent_center[0])**2 + 
                          (current_center[1] - adjacent_center[1])**2)
        
        if distance > max_distance:
            break
        
        result = frame_idx
    
    return result


def _apply_match(detection, obj_id, video_segment, frame_idx):
    """Apply match operation on a detection: mark as matched, copy mask, establish bidirectional link"""
    detection.is_matched = True
    detection.matched_track_id = obj_id
    detection.mask = deepcopy(video_segment.get_sam2_detection(frame_idx).mask)
    sam2_det = video_segment.get_sam2_detection(frame_idx)
    detection.matched_sam2_detection = sam2_det
    sam2_det.matched_unprocessed_detection = detection


def process_single_video(video_id, args, gpu_id, progress_manager, result_collector, gpu_load_manager):
    """Function to process a single video, used for multiprocessing"""
    try:
        # Set CUDA device
        torch.cuda.set_device(gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Build SAM2 predictor
        predictor = build_sam2_video_predictor(args.sam_config, args.sam_checkpoint, device=f"cuda:{gpu_id}")
        
        print(f"[GPU {gpu_id}] Starting to process video: {video_id}")
        
        # Create directory to save individual video results
        video_results_dir = os.path.join(args.output_dir, "video_results")
        os.makedirs(video_results_dir, exist_ok=True)
        
        # Check if already processed
        result_path = os.path.join(video_results_dir, f"{video_id}_result.pkl")
        if os.path.exists(result_path):
            print(f"[GPU {gpu_id}] Video {video_id} already has results, skipping")
            progress_manager.video_completed(video_id)
            return
        
        # Build video frame directory
        video_dir = os.path.join(args.dataset_root, args.split, f"SNGS-{video_id}", "img1")
        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"Video frame directory does not exist: {video_dir}")

        # Load video frames
        frame_names = load_video_frames(video_dir)
        num_frames = len(frame_names)
        
        if num_frames == 0:
            raise ValueError(f"No video frames found in {video_dir}")
        
        # Read the first frame
        first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
        if first_frame is None:
            raise ValueError(f"Cannot load the first frame {os.path.join(video_dir, frame_names[0])}")
            
        print(f"[GPU {gpu_id}] Video {video_id} first frame size: {first_frame.shape}")
        frame_height, frame_width = first_frame.shape[:2]
        
        # Load tracking data
        unprocessed_data_obj, img_id_to_frame_idx, frame_idx_2_image_id = load_tracklets_for_video(
            args.input_pklz, video_id, min_length=args.min_continuous_segment_length, num_frames=num_frames,
            min_propagate_box_dimension=args.min_propagate_box_dimension, split=args.split
        )
        
        # Initialize inference state
        inference_state = predictor.init_state(video_path=video_dir)
        predictor.reset_state(inference_state)
        
        # Convert unprocessed_data_obj to dict-based index while keeping references to original detection objects
        unprocessed_detections = defaultdict(list)
        for tracklet in unprocessed_data_obj.unprocessed_tracklets:
            for segment in tracklet.segments:
                for detection in segment.detections:
                    unprocessed_detections[detection.frame_idx].append(detection)
        
        # Count segment statistics
        total_segments = 0
        short_segments = 0
        for tracklet in unprocessed_data_obj.unprocessed_tracklets:
            for segment in tracklet.segments:
                total_segments += 1
                if len(segment.detections) > 0 and segment.detections[0].is_short_segment:
                    short_segments += 1
        
        valid_segments = total_segments - short_segments
        print(f"[GPU {gpu_id}] Video {video_id} total segments: {total_segments}")
        print(f"[GPU {gpu_id}] Video {video_id} short segments (length < {args.min_continuous_segment_length}): {short_segments}")
        print(f"[GPU {gpu_id}] Video {video_id} valid segments: {valid_segments}")
        
        # Store bbox prompts for each frame, used for rendering prompt video later
        frame_to_bbox_prompts = defaultdict(list)
        
        # Store unmatched segment information
        unmatched_segments = []
        
        # Use sam2_video_data class to replace the all_video_segments dict
        video_data = sam2_video_data()
        
        # Record statistics for each propagation
        propagate_stats = []
        
        while True:
            is_found, frame_idx, bbox_xyxy, track_id = get_next_detection(unprocessed_detections)
            if not is_found:
                break
            
            predictor.reset_state(inference_state)
            
            obj_id = int(track_id)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=np.array(bbox_xyxy)
            )
            
            frame_to_bbox_prompts[frame_idx].append((bbox_xyxy, obj_id))
            
            # Create a new video segment object
            video_segment = sam2_video_segment(obj_id)
            
            # Original dict format to keep the existing processing logic unchanged
            video_segments = {}
            
            with torch.autocast(device_type=f"cuda:{gpu_id}", dtype=torch.bfloat16):
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {}
                    for i, out_obj_id in enumerate(out_obj_ids):
                        mask_array = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze(0)
                        valid_mask = mask_array.any()
                        
                        seg_info = {
                            'mask': mask_array,
                            'valid_mask': valid_mask,
                            'has_outlier': False,
                            'bbox': bbox_from_mask(mask_array) if valid_mask else None,
                            'is_constrained': False
                        }
                        video_segments[out_frame_idx][out_obj_id] = seg_info
            
            # Initialize the continuous valid range starting from the prompt frame
            prompt_frame_idx = frame_idx
            
            # Check if the prompt frame itself is valid
            if not video_segments[prompt_frame_idx][obj_id]['valid_mask']:
                continue
            
            # Extend the continuous valid range forward/backward
            valid_range_start = prompt_frame_idx
            valid_range_end = prompt_frame_idx
            
            for check_idx in range(frame_idx - 1, -1, -1):
                if (check_idx in video_segments and obj_id in video_segments[check_idx] and 
                    video_segments[check_idx][obj_id]['valid_mask']):
                    valid_range_start = check_idx
                else:
                    break
            
            for check_idx in range(frame_idx + 1, num_frames):
                if (check_idx in video_segments and obj_id in video_segments[check_idx] and 
                    video_segments[check_idx][obj_id]['valid_mask']):
                    valid_range_end = check_idx
                else:
                    break
            
            # Detect and clean outlier masks only within the continuous valid range
            process_video_segments_in_range(video_segments, obj_id, prompt_frame_idx, valid_range_start, valid_range_end, args)
                
            # Distance constraint: use the extracted helper function
            valid_range_start2 = _search_valid_range_with_distance(
                video_segments, obj_id, prompt_frame_idx, valid_range_start, direction=-1)
            valid_range_end2 = _search_valid_range_with_distance(
                video_segments, obj_id, prompt_frame_idx, valid_range_end, direction=1)
                
            print(f"[GPU {gpu_id}] Video {video_id} object {obj_id} continuous valid range: {valid_range_start2} to {valid_range_end2}")
            
            # Add the processed masks to the sam2_video_segment object
            for frame_idx in range(valid_range_start2, valid_range_end2 + 1):
                if frame_idx in video_segments and obj_id in video_segments[frame_idx]:
                    seg_info = video_segments[frame_idx][obj_id]
                    detection_obj = sam2_detection(
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        mask=seg_info['mask'],
                        valid_mask=seg_info['valid_mask'],
                        has_outlier=seg_info['has_outlier'],
                        bbox=seg_info['bbox'],
                        is_constrained=seg_info['is_constrained']
                    )
                    video_segment.add_sam2_detection(detection_obj)
            
            video_data.add_segment(video_segment)
            
            # Record statistics for each propagation
            matched_count = 0
            segment_frames = list(range(valid_range_start2, valid_range_end2 + 1))
            start_unmatched_count = len(unmatched_segments)
            
            # Iterate over each frame within the continuous valid range, matching valid segmentation results with unprocessed detections
            for frame_idx in range(valid_range_start2, valid_range_end2 + 1):
                if frame_idx not in unprocessed_detections:
                    continue
                
                segment_bbox = video_segment.get_sam2_detection(frame_idx).bbox
                
                best_iou = 0.0
                best_seg_bbox_be_overlapped_ratio = 0.0
                best_iou_detection_idx = -1
                best_overlap_ratio_detection_idx = -1
                contatin_det_bbox_cnt = 0
                
                for i, detection in enumerate(unprocessed_detections[frame_idx]):
                    if not detection.is_matched:
                        iou, seg_bbox_be_overlapped_ratio, det_bbox_be_overlapped_ratio = calculate_iou(segment_bbox, detection.bbox_xyxy)
                        
                        if det_bbox_be_overlapped_ratio > 0.6:
                            contatin_det_bbox_cnt += 1
                        
                        if seg_bbox_be_overlapped_ratio > best_seg_bbox_be_overlapped_ratio:
                            best_seg_bbox_be_overlapped_ratio = seg_bbox_be_overlapped_ratio
                            best_overlap_ratio_detection_idx = i
                            
                        if iou > best_iou:
                            best_iou = iou
                            best_iou_detection_idx = i
                
                # Determine the best matching detection index
                matched_det_idx = None
                iou_ok = best_iou > args.best_iou_threshold
                overlap_ok = best_seg_bbox_be_overlapped_ratio > args.best_seg_bbox_be_overlapped_ratio_threshold
                
                if best_overlap_ratio_detection_idx == best_iou_detection_idx and best_overlap_ratio_detection_idx != -1 and (iou_ok or overlap_ok):
                    matched_det_idx = best_overlap_ratio_detection_idx
                elif best_iou_detection_idx != -1 and iou_ok:
                    matched_det_idx = best_iou_detection_idx
                elif best_overlap_ratio_detection_idx != -1 and overlap_ok:
                    matched_det_idx = best_overlap_ratio_detection_idx
                
                if matched_det_idx is not None:
                    _apply_match(unprocessed_detections[frame_idx][matched_det_idx], obj_id, video_segment, frame_idx)
                    matched_count += 1
                else:
                    # No match found, check if it is a missed detection (deduplication logic)
                    sam2_det = video_segment.get_sam2_detection(frame_idx)
                    current_mask = sam2_det.mask
                    current_obj_id = sam2_det.obj_id
                    is_duplicate = contatin_det_bbox_cnt > 1
                    
                    # Check overlap with already matched detections
                    if not is_duplicate:
                        for detection in unprocessed_detections[frame_idx]:
                            if detection.is_matched:
                                mask_iou, seg_mask_be_overlapped_ratio = calculate_mask_iou(current_mask, detection.mask)
                                if ((mask_iou > args.mask_iou_threshold or seg_mask_be_overlapped_ratio > args.seg_mask_be_overlapped_ratio_threshold) or 
                                    ((mask_iou > 0 or seg_mask_be_overlapped_ratio > 0) and current_obj_id == detection.matched_track_id)):
                                    is_duplicate = True
                                    break
                    
                    # Check overlap with existing unmatched_segments
                    if not is_duplicate:
                        for existing_segment in unmatched_segments:
                            if existing_segment.frame_idx == frame_idx:
                                mask_iou, seg_mask_be_overlapped_ratio = calculate_mask_iou(current_mask, existing_segment.mask)
                                if ((mask_iou > args.mask_iou_threshold or seg_mask_be_overlapped_ratio > args.seg_mask_be_overlapped_ratio_threshold) or 
                                    ((mask_iou > 0 or seg_mask_be_overlapped_ratio > 0) and current_obj_id == existing_segment.obj_id)):
                                    is_duplicate = True
                                    break
                    
                    if not is_duplicate:
                        unmatched_seg = unmatched_segment.from_sam2_detection(sam2_det)
                        sam2_det.set_unmatched_segment(unmatched_seg)
                        unmatched_segments.append(unmatched_seg)
            
            # Add the statistics for this propagation to the list
            actually_added_unmatched = len(unmatched_segments) - start_unmatched_count
            
            propagate_stats.append({
                'track_id': track_id,
                'obj_id': obj_id,
                'prompt_frame': prompt_frame_idx,
                'frame_range': (valid_range_start2, valid_range_end2),
                'total_frames': len(segment_frames),
                'matched_bboxes': matched_count,
                'unmatched_segments_added': actually_added_unmatched
            })
            
            print(f"[GPU {gpu_id}] Propagation stats - Video {video_id} Track ID: {track_id}, Obj ID: {obj_id}")
            print(f"[GPU {gpu_id}]   Range: {valid_range_start2}-{valid_range_end2}, total {len(segment_frames)} frames")
            print(f"[GPU {gpu_id}]   Successfully matched: {matched_count} bboxes")
            print(f"[GPU {gpu_id}]   Added: {actually_added_unmatched} unmatched segments")
            print(f"[GPU {gpu_id}]   Match rate: {matched_count/len(segment_frames):.2f}, New segment rate: {actually_added_unmatched/len(segment_frames):.2f}")
        
        # Summarize all propagation statistics for this video
        print(f"\n[GPU {gpu_id}] Propagation statistics summary for video {video_id}:")
        total_propagate_frames = sum(stat['total_frames'] for stat in propagate_stats)
        total_matched_bboxes = sum(stat['matched_bboxes'] for stat in propagate_stats)
        total_unmatched_added = sum(stat['unmatched_segments_added'] for stat in propagate_stats)
        
        if total_propagate_frames > 0:
            print(f"[GPU {gpu_id}] Total {len(propagate_stats)} propagations, covering {total_propagate_frames} frames")
            print(f"[GPU {gpu_id}] Total matched: {total_matched_bboxes} bboxes (match rate: {total_matched_bboxes/total_propagate_frames:.2f})")
            print(f"[GPU {gpu_id}] Total added: {total_unmatched_added} unmatched segments (new segment rate: {total_unmatched_added/total_propagate_frames:.2f})")
        
        print(f"[GPU {gpu_id}] Video {video_id} found {len(unmatched_segments)} unmatched segments")
        for i, segment in enumerate(unmatched_segments[:5]):
            print(f"[GPU {gpu_id}] Unmatched segment #{i}: frame {segment.frame_idx}, obj ID {segment.obj_id}, bbox {segment.bbox}")
        
        # Create and count tracklets
        sam2_tracklets = video_data.create_tracklets()
        print(f"[GPU {gpu_id}] Video {video_id} created {len(sam2_tracklets)} tracklets")
        
        # Iterate over all frames, find unmatched but valid on-pitch detections
        valid_unmatched_detections = [
            {'frame_idx': fidx, 'bbox_xyxy': det.bbox_xyxy}
            for fidx in unprocessed_detections
            for det in unprocessed_detections[fidx]
            if not det.is_matched
        ]
        print(f"[GPU {gpu_id}] Video {video_id} found {len(valid_unmatched_detections)} unmatched detections")

        # If fix duplicate track_ids is enabled, perform the fix
        if args.fix_duplicate_track_ids:
            fix_duplicate_track_ids_in_data(unprocessed_data_obj, unmatched_segments, video_data, video_id)

        # Generate visualization only when requested
        if args.visualize_video:
            _generate_visualization(
                args, video_id, gpu_id, video_dir, frame_names, 
                frame_width, frame_height, frame_to_bbox_prompts,
                unprocessed_detections, unmatched_segments, video_data
            )
            
        # Clear mask data to save memory
        _clear_masks(unprocessed_detections, unmatched_segments, unprocessed_data_obj, video_data)
        
        # Single video result
        video_result = {
            'unmatched_segments': unmatched_segments,
            'unprocessed_data': unprocessed_data_obj
        }
        
        if args.fix_duplicate_track_ids:
            video_result['track_ids_fixed'] = True
        
        # Break circular references in the data structure for safe serialization
        video_result = break_circular_references(video_result)
        
        # Immediately write the single video result to a pkl file
        video_result_path = os.path.join(video_results_dir, f"{video_id}_result.pkl")
        with open(video_result_path, 'wb') as f:
            pickle.dump(video_result, f)
        print(f"[GPU {gpu_id}] Video {video_id} results saved to: {video_result_path}")
        
        # Collect results
        result_collector.add_result(video_id, video_result)
        
        # Update progress
        progress_manager.video_completed(video_id)
        
    except Exception as e:
        error_msg = f"Error processing video {video_id}: {str(e)}\n{traceback.format_exc()}"
        print(f"[GPU {gpu_id}] {error_msg}")
        progress_manager.video_failed(video_id, error_msg)
    finally:
        # Clean up GPU memory and release GPU load
        torch.cuda.empty_cache()
        gpu_load_manager.release_gpu(gpu_id)


def _clear_masks(unprocessed_detections, unmatched_segments, unprocessed_data_obj, video_data):
    """Clear mask data from all data structures to save memory"""
    for frame_idx in unprocessed_detections:
        for detection in unprocessed_detections[frame_idx]:
            detection.mask = None
    for segment in unmatched_segments:
        segment.mask = None
    for tracklet in unprocessed_data_obj.unprocessed_tracklets:
        for segment in tracklet.segments:
            for detection in segment.detections:
                detection.mask = None
    for segment in video_data.segments:
        for frame_idx, detection in segment.detections.items():
            detection.mask = None


def _generate_visualization(args, video_id, gpu_id, video_dir, frame_names,
                            frame_width, frame_height, frame_to_bbox_prompts,
                            unprocessed_detections, unmatched_segments, video_data):
    """Generate visualization videos"""
    vis_output_dir = args.output_dir
    os.makedirs(vis_output_dir, exist_ok=True)
    
    fps = args.fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    video_paths = {
        'prompts': os.path.join(vis_output_dir, f'{video_id}_prompts_video.mp4'),
        'detections_segments': os.path.join(vis_output_dir, f'{video_id}_detections_segments_video.mp4'),
        'all_segments': os.path.join(vis_output_dir, f'{video_id}_all_segments_video.mp4'),
    }
    
    writers = {
        name: cv2.VideoWriter(path, fourcc, fps, (frame_width, frame_height))
        for name, path in video_paths.items()
    }
    
    print(f"[GPU {gpu_id}] Starting video generation...")
    for frame_idx in tqdm(range(len(frame_names)), desc=f"[GPU {gpu_id}] Rendering video frames - {video_id}"):
        frame_path = os.path.join(video_dir, frame_names[frame_idx])
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"[GPU {gpu_id}] Warning: Cannot load frame {frame_path}")
            continue
        
        frame_counter_text = f"Frame: {frame_idx+1}/{len(frame_names)}"
        text_size = cv2.getTextSize(frame_counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_pos = (frame_width - text_size[0] - 10, 30)
        
        # 1. Render prompts video
        prompts_frame = frame.copy()
        cv2.putText(prompts_frame, frame_counter_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if frame_idx in frame_to_bbox_prompts:
            for bbox, obj_id in frame_to_bbox_prompts[frame_idx]:
                prompts_frame = draw_box_with_id(prompts_frame, bbox, obj_id)
        writers['prompts'].write(prompts_frame)
        
        # 2. Render detections and unmatched_segments video
        det_seg_frame = frame.copy()
        cv2.putText(det_seg_frame, frame_counter_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if frame_idx in unprocessed_detections:
            for detection in unprocessed_detections[frame_idx]:
                bbox = detection.bbox_xyxy
                if detection.is_matched:
                    color = (0, 255, 0)
                    text = f"ID: {int(detection.track_id)}"
                    matched_text = f"M: {int(detection.matched_track_id)}"
                    if detection.matched_track_id >= 100:
                        matched_text += " (fixed)"
                else:
                    color = (0, 0, 255)
                    text = f"ID: {int(detection.track_id)} unmatched"
                
                cv2.rectangle(det_seg_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(det_seg_frame, text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if detection.is_matched:
                    cv2.putText(det_seg_frame, matched_text, (int(bbox[0]), int(bbox[1]) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        for segment in unmatched_segments:
            if segment.frame_idx == frame_idx and segment.valid_mask:
                det_seg_frame = apply_mask_to_image(
                    det_seg_frame, segment.mask, 
                    obj_id=segment.obj_id, alpha=0.3, add_contour=True,
                    is_constrained=segment.is_constrained,
                    has_outlier=segment.has_outlier
                )
                
                if segment.obj_id >= 100:
                    seg_bbox = bbox_from_mask(segment.mask)
                    if seg_bbox is not None:
                        cv2.putText(det_seg_frame, "fixed",
                                (int(seg_bbox[0]), int(seg_bbox[1]) - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        writers['detections_segments'].write(det_seg_frame)
        
        # 3. Render all_video_segments video
        all_seg_frame = frame.copy()
        cv2.putText(all_seg_frame, frame_counter_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        for detection in video_data.get_all_detections_in_frame(frame_idx):
            if detection.valid_mask:
                all_seg_frame = apply_mask_to_image(
                    all_seg_frame, detection.mask, 
                    obj_id=detection.obj_id, alpha=0.3, add_contour=True,
                    is_constrained=detection.is_constrained,
                    has_outlier=detection.has_outlier
                )
        
        writers['all_segments'].write(all_seg_frame)
    
    # Release resources
    for writer in writers.values():
        writer.release()
    
    print(f"[GPU {gpu_id}] Video {video_id} saved to:")
    for name, path in video_paths.items():
        print(f"[GPU {gpu_id}]   - {name}: {path}")


def check_video_ids_in_pklz(pklz_path, video_ids):
    """Check whether the pklz file contains all video_ids to be processed"""
    try:
        with zipfile.ZipFile(pklz_path) as zf:
            pklz_video_ids = {
                f.split('.')[0] for f in zf.namelist() 
                if f.endswith('.pkl') and 'image' not in f
            }
            
            print(f"The pklz file contains {len(pklz_video_ids)} video IDs")
            print(f"First 10 video ID examples: {sorted(pklz_video_ids)[:10]}")
            
            available = [vid for vid in video_ids if vid in pklz_video_ids]
            missing = [vid for vid in video_ids if vid not in pklz_video_ids]
            
            return available, missing
            
    except Exception as e:
        raise ValueError(f"Cannot read pklz file {pklz_path}: {str(e)}")


def process_video_wrapper(args_tuple):
    """Wrapper function for multiprocessing pool, dynamically assigns GPU"""
    video_id, progress_manager, result_collector, gpu_load_manager, global_args = args_tuple
    
    # Wait to acquire an available GPU
    gpu_id = None
    while gpu_id is None:
        gpu_id = gpu_load_manager.get_best_gpu()
        if gpu_id is None:
            time.sleep(1)
    
    print(f"Video {video_id} assigned to GPU {gpu_id}")
    process_single_video(video_id, global_args, gpu_id, progress_manager, result_collector, gpu_load_manager)


def _load_all_video_results(video_results_dir, video_ids):
    """Load results of all processed videos"""
    all_results = {}
    for video_id in video_ids:
        result_path = os.path.join(video_results_dir, f"{video_id}_result.pkl")
        if os.path.exists(result_path):
            with open(result_path, 'rb') as f:
                all_results[video_id] = pickle.load(f)
    return all_results


def main():
    args = parse_args()
    
    # Parse GPU list
    gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_list.split(',')]
    print(f"Using GPUs: {gpu_ids}")
    print(f"Max processes per GPU: {args.max_processes_per_gpu}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    video_results_dir = os.path.join(args.output_dir, "video_results")
    os.makedirs(video_results_dir, exist_ok=True)
    
    # Load metadata and determine the list of video IDs to process
    video_ids = []
    if args.video_id_list:
        video_ids = [vid.strip() for vid in args.video_id_list.split(',')]
    elif args.video_id_start is not None and args.video_id_end is not None:
        if args.video_id_start > args.video_id_end:
            raise ValueError(f"video_id_start ({args.video_id_start}) cannot be greater than video_id_end ({args.video_id_end})")
        video_ids = [str(vid) for vid in range(args.video_id_start, args.video_id_end + 1)]
        print(f"Generated video ID list from range: {args.video_id_start}-{args.video_id_end}, total {len(video_ids)} videos")
    elif args.video_id_start is not None or args.video_id_end is not None:
        raise ValueError("Both --video_id_start and --video_id_end must be specified together")
    elif args.dataset_root:
        try:
            with open(os.path.join(args.dataset_root, "sequences_info.json"), 'r') as f:
                metadata = json.load(f)
                
            split_key = 'validation' if args.split == 'valid' else args.split
            video_ids = [vid["name"].split('-')[1] for vid in metadata[split_key]]
            print(f"Found {len(video_ids)} videos from metadata")
        except Exception as e:
            print(f"Cannot load metadata: {e}")
            if args.video_id_list is None:
                raise ValueError("Must provide --video_id_list or a valid --metadata_path")
    
    # Check already processed videos to avoid redundant computation
    already_processed_videos = []
    for video_id in video_ids[:]:
        result_path = os.path.join(video_results_dir, f"{video_id}_result.pkl")
        if os.path.exists(result_path):
            print(f"Video {video_id} already has results, skipping")
            already_processed_videos.append(video_id)
            video_ids.remove(video_id)
    
    # Before starting processing, check if the pklz file contains all video_ids to be processed
    if video_ids:
        print("\nChecking video IDs in pklz file...")
        available_video_ids, missing_video_ids = check_video_ids_in_pklz(args.input_pklz, video_ids)
        
        if missing_video_ids:
            print(f"\n⚠️  Warning: The following {len(missing_video_ids)} video IDs do not exist in the pklz file:")
            for missing_id in missing_video_ids:
                print(f"  - {missing_id}")
            
            print(f"\n✓ Found {len(available_video_ids)} video IDs in the pklz file")
            video_ids = available_video_ids
            print(f"Will continue processing {len(video_ids)} videos that exist in the pklz file")
        else:
            print(f"✓ All {len(video_ids)} video IDs to be processed exist in the pklz file")
    
    if not video_ids:
        print("All videos have been processed, no need to recompute")
        if not (args.save_results or args.save_refined_pklz):
            return
        processed_videos = already_processed_videos
    else:
        print(f"Need to process {len(video_ids)} videos")
        processed_videos = already_processed_videos
    
    if video_ids:
        # Initialize multiprocessing manager
        manager = Manager()
        progress_manager = ProcessSafeProgressManager(manager, len(video_ids))
        result_collector = ProcessSafeResultCollector(manager)
        gpu_load_manager = ProcessSafeGPULoadManager(manager, gpu_ids, args.max_processes_per_gpu)
        
        total_processes = len(gpu_ids) * args.max_processes_per_gpu
        print(f"Using {total_processes} processes to handle {len(video_ids)} videos")
        
        tasks = [
            (video_id, progress_manager, result_collector, gpu_load_manager, args)
            for video_id in video_ids
        ]
        
        start_time = time.time()
        
        with Pool(processes=total_processes) as pool:
            try:
                for _ in tqdm(pool.imap_unordered(process_video_wrapper, tasks), total=len(tasks), desc="Video processing progress"):
                    pass
            except KeyboardInterrupt:
                print("\nInterrupt signal received, stopping all processes...")
                pool.terminate()
                pool.join()
                raise
        
        processing_time = time.time() - start_time
        
        completed, failed, total = progress_manager.get_progress()
        failed_ids = {fail[0] for fail in progress_manager.failed_videos}
        processed_videos.extend([vid for vid in video_ids if vid not in failed_ids])
        
        print(f"\nProcessing complete!")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print(f"Successfully processed: {completed} videos")
        print(f"Failed: {failed} videos")
        if failed > 0:
            print("Failed videos:")
            for video_id, error in progress_manager.failed_videos:
                print(f"  - {video_id}: {error}")
    
    # Save results (save_results and save_refined_pklz share a single load)
    if args.save_results or args.save_refined_pklz:
        all_results = _load_all_video_results(video_results_dir, processed_videos)
        
        if args.save_results:
            output_pkl_path = args.output_pkl or os.path.join(args.output_dir, "results.pkl")
            print(f"Aggregating all video results to: {output_pkl_path}")
            with open(output_pkl_path, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"Aggregated results saved to: {output_pkl_path}")
        
        if args.save_refined_pklz:
            refined_pklz_path = generate_refined_pklz(args, all_results)
            print(f"Refined pklz file saved to: {refined_pklz_path}")
    
    print("All video processing complete")


if __name__ == "__main__":
    if os.name == 'nt':
        multiprocessing.freeze_support()
    main()
