import os
import numpy as np
import torch
import cv2
import zipfile
import pickle
import pandas as pd
from collections import Counter
from copy import deepcopy
from tqdm import tqdm

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PITCH_X_MARGIN = 10.0
PITCH_Y_MARGIN = 5.0
COORD_X_MIN = -((PITCH_LENGTH / 2) + PITCH_X_MARGIN)  # -62.5
COORD_X_MAX = ((PITCH_LENGTH / 2) + PITCH_X_MARGIN)   # 62.5
COORD_Y_MIN = -((PITCH_WIDTH / 2) + PITCH_Y_MARGIN)   # -39.0
COORD_Y_MAX = ((PITCH_WIDTH / 2) + PITCH_Y_MARGIN)    # 39.0

###################
# Device configuration
###################
def setup_device():
    """Set up and return the compute device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Use bfloat16 for CUDA devices
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # Enable tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    return device

###################
# Visualization helper functions
###################
def apply_mask_to_image(image, mask, obj_id=None, random_color=False, alpha=0.5, add_contour=True, is_constrained=False, has_outlier=False):
    """
    Apply a mask to an image, optionally adding contours
    
    Args:
        image: Original image (H,W,3)
        mask: Mask array (H,W)
        obj_id: Object ID used to determine color
        random_color: Whether to use a random color
        alpha: Blending transparency
        add_contour: Whether to add contours
        is_constrained: Whether this is a constrained mask
        has_outlier: Whether this is an outlier mask
    
    Returns:
        Image with mask applied (H,W,3)
    """
    # Set color (OpenCV uses BGR)
    if random_color:
        color_rgb = np.random.random(3) * 255
    else:
        colors = [
            # Basic colors
            [255, 0, 0],     # Red
            [0, 255, 0],     # Green
            [0, 0, 255],     # Blue
            [255, 255, 0],   # Cyan
            [255, 0, 255],   # Magenta
            [0, 255, 255],   # Yellow
            # Red shades with different brightness
            [128, 0, 0],     # Dark red
            [255, 128, 128], # Light red
            [255, 102, 102], # Pink red
            [153, 0, 0],     # Maroon
            # Green shades with different brightness
            [0, 128, 0],     # Dark green
            [128, 255, 128], # Light green
            [0, 153, 0],     # Forest green
            [102, 255, 102], # Bright green
            # Blue shades with different brightness
            [0, 0, 128],     # Dark blue
            [128, 128, 255], # Light blue
            [0, 102, 204],   # Royal blue
            [51, 153, 255],  # Sky blue
            # Mixed colors
            [128, 128, 0],   # Olive
            [128, 0, 128],   # Purple
            [0, 128, 128],   # Teal
            [204, 204, 0],   # Mustard
            [204, 0, 204],   # Fuchsia
            [0, 204, 204],   # Turquoise
            # Gray shades
            [192, 192, 192], # Silver
            [128, 128, 128], # Gray
            [64, 64, 64],    # Dark gray
            # Brown shades
            [153, 102, 0],   # Brown
            [204, 153, 102], # Light brown
            [102, 51, 0],    # Dark brown
            # Other mixed colors
            [153, 153, 255], # Lavender
            [255, 153, 153], # Peach
            [255, 204, 153], # Apricot
            [204, 255, 153], # Lime green
            [153, 255, 204], # Mint
            [153, 204, 255], # Pale blue
            [204, 153, 255], # Light purple
            [255, 153, 204]  # Pink
        ]
        cmap_idx = 0 if obj_id is None else (obj_id % len(colors))
        color_rgb = colors[cmap_idx]
    
    # Convert RGB to BGR (OpenCV format)
    color_bgr = [color_rgb[2], color_rgb[1], color_rgb[0]]
    
    # Create boolean representation of the mask
    h, w = mask.shape[-2:]
    binary_mask = mask.reshape(h, w).astype(bool)
    
    # Create a copy of the output image
    output = image.copy()
    
    # Apply color to the mask region
    if np.any(binary_mask):
        for c in range(3):
            output[:, :, c] = np.where(
                binary_mask, 
                output[:, :, c] * (1 - alpha) + color_bgr[c] * alpha,
                output[:, :, c]
            )
    
    # Add contours
    if add_contour:
        # Convert mask to uint8 type
        mask_uint8 = binary_mask.astype(np.uint8) * 255
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours
        cv2.drawContours(output, contours, -1, color_bgr, 2)
    
    # If there are non-zero regions, compute bounding box and add object ID label
    if np.any(binary_mask):
        # Find coordinates of non-zero regions
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Draw bounding box
        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), color_bgr, 2)
        
        # Prepare text labels to display
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Set up label list
        labels = []
        if obj_id is not None:
            labels.append(f"ID: {obj_id}")
        if is_constrained:
            labels.append("constrained")
        if has_outlier:
            labels.append("outlier")
        
        # Draw all labels
        for i, label in enumerate(labels):
            # Determine text position (below the previous label)
            text_x = x_min
            text_y = y_max + 20 + (i * 15)  # 15 pixels spacing between labels
            
            # Get text size to adjust background rectangle
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Add background rectangle for text readability
            cv2.rectangle(output, 
                          (text_x, text_y - text_height), 
                          (text_x + text_width, text_y + 5), 
                          (0, 0, 0), 
                          -1)  # -1 means filled rectangle
            
            # Add text
            cv2.putText(output, 
                        label, 
                        (text_x, text_y), 
                        font, 
                        font_scale, 
                        (255, 255, 255),  # White text
                        font_thickness)
    
    return output

def draw_box_with_id(image, box, obj_id):
    """
    Draw a bounding box with an object ID on an image
    
    Args:
        image: Original image (H,W,3)
        box: Bounding box coordinates [x0, y0, x1, y1]
        obj_id: Object ID
    
    Returns:
        Image with bounding box and ID
    """
    # Copy image to avoid modifying the original
    output = image.copy()
    
    # Set color (OpenCV uses BGR)
    colors = [
        [255, 0, 0],     # Red
        [0, 255, 0],     # Green
        [0, 0, 255],     # Blue
        [255, 255, 0],   # Cyan
        [255, 0, 255],   # Magenta
        [0, 255, 255],   # Yellow
        [128, 0, 0],     # Dark red
        [0, 128, 0],     # Dark green
        [0, 0, 128],     # Dark blue
        [128, 128, 0],   # Olive
        [128, 0, 128],   # Purple
        [0, 128, 128],   # Teal
    ]
    cmap_idx = obj_id % len(colors)
    color_rgb = colors[cmap_idx]
    color_bgr = [color_rgb[2], color_rgb[1], color_rgb[0]]
    
    # Draw bounding box
    x0, y0, x1, y1 = map(int, box)
    cv2.rectangle(output, (x0, y0), (x1, y1), color=color_bgr, thickness=2)
    
    # Add object ID label
    label = f"ID: {obj_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Add background rectangle for text readability
    text_x = x0
    text_y = y0 - 10 if y0 - 10 > 10 else y1 + 20  # Above or below the bounding box
    
    cv2.rectangle(output, 
                  (text_x, text_y - text_height), 
                  (text_x + text_width, text_y + 5), 
                  (0, 0, 0), 
                  -1)  # -1 means filled rectangle
    
    # Add text
    cv2.putText(output, 
                label, 
                (text_x, text_y), 
                font, 
                font_scale, 
                (255, 255, 255),  # White text
                font_thickness)
    
    return output

def load_tracklets_for_video(pklz_path, video_id, min_length=3, num_frames=None, min_propagate_box_dimension=0, split='challenge'):
    """
    Load tracking data for a specific video from a pklz file and process it, returning an unprocessed_data object
    
    Args:
        pklz_path: Path to the pklz file
        video_id: Video ID to load
        min_length: Minimum tracklet length; tracklets shorter than this will be filtered out
        num_frames: Total number of frames in the video, used to filter detections out of range
        min_propagate_box_dimension: Minimum bounding box dimension requirement; if width or height is less than this, propagation is skipped
        split: Dataset split, 'train', 'valid', 'test' or 'challenge', used to determine the image_id prefix
    
    Returns:
        unprocessed_data object containing all processed unprocessed_tracklets
        image_id_to_frame_idx mapping
        frame_idx_2_image_id mapping
    """
    # Load pklz data
    with zipfile.ZipFile(pklz_path) as zf:
        pkl_files = zf.namelist()
        
        # Find the pkl file for the specified video
        video_pkl = f"{video_id}.pkl"
        if video_pkl not in pkl_files:
            raise ValueError(f"Video {video_id} not found in pklz file")
        
        # Load video data
        with zf.open(video_pkl) as fp:
            data = pickle.load(fp)

    # Convert to DataFrame for easier processing
    df = pd.DataFrame(data)

    # Group by track_id
    raw_tracklets = {}
    for track_id, group in df.groupby('track_id'):
        raw_tracklets[track_id] = group
    print(f"Found {len(raw_tracklets)} tracklets")

    # Determine prefix based on split: train=1, valid=2, test=3, challenge=4, franz=5, sn500=6
    split_prefix = {'train': '1', 'valid': '2', 'test': '3', 'challenge': '4', 'franz': '5', 'sn500': '6'}[split]
    frame_idx_2_image_id = {i: f'{split_prefix}{video_id}{(i+1):06d}' for i in range(num_frames)}
    img_id_to_frame_idx = {img_id: i for i, img_id in frame_idx_2_image_id.items()}
    
    # Convert raw_tracklets to a list of unprocessed_tracklets
    unprocessed_tracklets = []
    
    for track_id, tracklet_df in raw_tracklets.items():
        # Create a new tracklet object
        tracklet = unprocessed_tracklet(track_id)
        
        # Sort by frame_idx to check continuity
        sorted_frames = []
        for image_id, detection_idx, bbox_ltwh in zip(tracklet_df['image_id'], tracklet_df.index.values, tracklet_df['bbox_ltwh']):
            frame_idx = img_id_to_frame_idx[image_id]
            # If num_frames is specified, filter detections out of range
            if num_frames is not None and frame_idx >= num_frames:
                continue
            sorted_frames.append((frame_idx, detection_idx, bbox_ltwh))
        
        # If no valid frames, skip this tracklet
        if not sorted_frames:
            continue
            
        sorted_frames.sort(key=lambda x: x[0])
        
        # Split into continuous segments
        current_segment = None
        
        for i, (frame_idx, detection_idx, bbox_ltwh) in enumerate(sorted_frames):
            # Check if a new segment needs to be created
            if current_segment is None or frame_idx > (sorted_frames[i-1][0] + 1):
                # If current segment is not empty, add it to the tracklet
                if current_segment is not None:
                    tracklet.add_segment(current_segment)
                
                # Create a new segment
                current_segment = unprocessed_segment(track_id)
            
            # Add current detection to the current segment
            bbox_xyxy = np.array([bbox_ltwh[0], bbox_ltwh[1], bbox_ltwh[0] + bbox_ltwh[2], bbox_ltwh[1] + bbox_ltwh[3]])

            # Create detection object
            detection = unprocessed_detection(
                detection_idx=detection_idx,
                frame_idx=frame_idx,
                bbox_ltwh=bbox_ltwh,
                bbox_xyxy=bbox_xyxy,
                track_id=track_id,
                is_matched=False,
                matched_track_id=None,
                tried_propagation=False,
                segment_length=0,  # Temporarily set to 0, updated later
                is_short_segment=False  # Temporarily set to False, updated later
            )
            
            # Check if bounding box dimensions meet the minimum size requirement
            width = bbox_ltwh[2]
            height = bbox_ltwh[3]
            box_dimension_valid = width >= min_propagate_box_dimension and height >= min_propagate_box_dimension
            if not box_dimension_valid:
                detection.tried_propagation = True
            
            current_segment.add_detection(detection)
        
        # Add the last segment
        if current_segment is not None:
            tracklet.add_segment(current_segment)
        
        # Update segment_length and is_short_segment for detections in each segment
        for segment in tracklet.segments:
            segment_length = len(segment.detections)
            is_short_segment = segment_length < min_length
            
            for detection in segment.detections:
                detection.segment_length = segment_length
                detection.is_short_segment = is_short_segment
                # For short segments, set tried_propagation to True to skip them
                if is_short_segment:
                    detection.tried_propagation = True
        
        # Only add the tracklet to the list if it has segments
        if tracklet.segments:
            unprocessed_tracklets.append(tracklet)
    
    # Create and return the unprocessed_data object
    return unprocessed_data(unprocessed_tracklets), img_id_to_frame_idx, frame_idx_2_image_id

def load_video_frames(video_dir):
    """
    Load video frames
    
    Args:
        video_dir: Directory containing video frames
    
    Returns:
        Sorted list of frame names
    """
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return frame_names 

def get_next_detection(unprocessed_detections):
    # Iterate over all frames
    for frame_idx in sorted(unprocessed_detections.keys()):
        # Iterate over all detections in this frame
        for detection in unprocessed_detections[frame_idx]:
            # Find the first unmatched detection
            if not detection.is_matched and not detection.tried_propagation:
                # Mark as propagation attempted
                detection.tried_propagation = True
                
                track_id = detection.track_id
                bbox_xyxy = detection.bbox_xyxy
                
                # Look back 50 frames for match history of the same track_id
                look_back_frames = 50
                matched_history = []
                
                # Calculate start frame index (not less than 0)
                start_frame = max(0, frame_idx - look_back_frames)
                
                # Collect matched_track_id from previous frames with the same track_id
                for prev_frame_idx in range(start_frame, frame_idx):
                    if prev_frame_idx in unprocessed_detections:
                        for prev_detection in unprocessed_detections[prev_frame_idx]:
                            if (prev_detection.track_id == track_id and 
                                prev_detection.is_matched and 
                                prev_detection.matched_track_id is not None):
                                matched_history.append(prev_detection.matched_track_id)
                
                # If match history count is greater than 5
                if len(matched_history) > 5:
                    # Count occurrences of each matched_track_id
                    counter = Counter(matched_history)
                    # Find the most common matched_track_id and its count
                    most_common_id, most_common_count = counter.most_common(1)[0]
                    
                    # If more than two-thirds of matches point to the same matched_track_id
                    if most_common_count > (len(matched_history) * 2 / 3):
                        print(f"Detected match history for track_id {track_id} at frame {frame_idx}: {most_common_count}/{len(matched_history)} -> {most_common_id}")
                        # Return this matched_track_id as obj_id instead of the original track_id
                        return True, frame_idx, bbox_xyxy, most_common_id
                
                # If not enough match history or no clear majority match, use original track_id
                return True, frame_idx, bbox_xyxy, track_id

    return False, None, None, None

# Calculate IoU and overlap ratios between two bounding boxes
def calculate_iou(box1, box2):
    # Ensure both box1 and box2 are not None
    if box1 is None or box2 is None:
        return 0.0, 0.0, 0.0
    
    # Calculate intersection region
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x2 < x1 or y2 < y1:
        return 0.0, 0.0, 0.0
    
    # Calculate intersection area
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate areas of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    # Calculate overlap ratio relative to box1 area
    box1_be_overlapped_ratio = intersection_area / box1_area if box1_area > 0 else 0.0
    
    # Calculate overlap ratio relative to box2 area
    box2_be_overlapped_ratio = intersection_area / box2_area if box2_area > 0 else 0.0
    
    return iou, box1_be_overlapped_ratio, box2_be_overlapped_ratio

# Calculate IoU between two masks
def calculate_mask_iou(mask1, mask2):
    """
    Calculate IoU between two binary masks
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
    
    Returns:
        mask_iou: IoU value of the masks
    """
    if mask1 is None or mask2 is None:
        return 0.0
    
    # Ensure masks are boolean type
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    
    # Calculate overlap ratio relative to mask1 area
    seg_mask_be_overlapped_ratio = intersection / mask1.sum() if mask1.sum() > 0 else 0.0
    
    return iou, seg_mask_be_overlapped_ratio
    
# Get the center point of a bounding box
def get_bbox_center(bbox):
    if bbox is None:
        return None
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

# Detect whether a bounding box is an outlier
def detect_outlier_bbox(video_segments, frame_idx, obj_id, window_size=2, ratio_threshold=1.4):
    """
    Detect whether the bounding box in the current frame is an outlier by comparing
    its size with the average size of preceding frames
    
    Args:
        video_segments: Video segmentation results dictionary
        frame_idx: Current frame index
        obj_id: Object ID
        window_size: Number of preceding frames to consider
        ratio_threshold: Width/height ratio threshold; exceeding this is considered an outlier
        
    Returns:
        Boolean, True indicates an outlier, False indicates normal
    """
    # Check if the object exists in the current frame
    if frame_idx not in video_segments or obj_id not in video_segments[frame_idx]:
        return False
    
    # Get current bounding box
    current_bbox = video_segments[frame_idx][obj_id]['bbox']
    if current_bbox is None:
        return False
    
    # Get width and height of the current bounding box
    current_width = current_bbox[2] - current_bbox[0]
    current_height = current_bbox[3] - current_bbox[1]
    
    # Add fixed size threshold check
    if current_width + current_height > 500:
        return True
    if current_width + current_height < 150:
        return False
    
    # Collect bounding box widths and heights from preceding window_size frames
    prev_widths = []
    prev_heights = []
    
    # Only consider the preceding window_size frames
    for i in range(-window_size, 0):
        check_frame_idx = frame_idx + i
        
        if (check_frame_idx in video_segments and 
            obj_id in video_segments[check_frame_idx] and 
            video_segments[check_frame_idx][obj_id]['valid_mask'] and
            video_segments[check_frame_idx][obj_id]['bbox'] is not None):
            
            bbox = video_segments[check_frame_idx][obj_id]['bbox']
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            prev_widths.append(bbox_width)
            prev_heights.append(bbox_height)
        else:
            break
    
    # If not enough preceding frames, cannot determine if it's an outlier
    if len(prev_widths) < 1:
        return False
    
    # Calculate average width and height of preceding frames' bounding boxes
    avg_width = np.mean(prev_widths)
    avg_height = np.mean(prev_heights)
    
    # Calculate ratio of current bounding box to average width and height
    width_ratio = current_width / avg_width if avg_width > 0 else 1.0
    height_ratio = current_height / avg_height if avg_height > 0 else 1.0
    
    # If width or height ratio exceeds the threshold, it's an outlier
    return width_ratio > ratio_threshold or height_ratio > ratio_threshold

# Constrain current frame's bbox size based on previous frame's bbox
def constrain_bbox_with_previous(mask, current_bbox, prev_bbox, max_expansion_ratio=1.0, max_width_offset=30, max_height_offset=60):
    """
    Constrain current frame's bbox size based on the previous frame's bbox to prevent sudden expansion
    
    Args:
        mask: Current frame's mask
        current_bbox: Current frame's bbox [x_min, y_min, x_max, y_max]
        prev_bbox: Previous frame's bbox [x_min, y_min, x_max, y_max]
        max_expansion_ratio: Maximum allowed expansion ratio
        max_width_offset: Maximum width offset
        max_height_offset: Maximum height offset
        
    Returns:
        constrained_mask: Constrained mask
        constrained_bbox: Constrained bbox
        is_constrained: Whether constraining was applied
    """
    # If current bbox width+height sum < 125 or previous bbox is None, skip constraining
    if prev_bbox is None:
        return mask, current_bbox, False
    
    current_width = current_bbox[2] - current_bbox[0]
    current_height = current_bbox[3] - current_bbox[1]
    
    if current_width + current_height < 125:
        return mask, current_bbox, False
    
    # Calculate center and dimensions of the previous frame's bbox
    prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
    prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
    prev_width = prev_bbox[2] - prev_bbox[0]
    prev_height = prev_bbox[3] - prev_bbox[1]
    
    # Calculate maximum allowed dimensions
    max_width = prev_width * max_expansion_ratio + max_width_offset
    max_height = prev_height * max_expansion_ratio + max_height_offset
    
    # Calculate constrained bbox
    constrained_x_min = max(current_bbox[0], prev_center_x - max_width / 2)
    constrained_y_min = max(current_bbox[1], prev_center_y - max_height / 2)
    constrained_x_max = min(current_bbox[2], prev_center_x + max_width / 2)
    constrained_y_max = min(current_bbox[3], prev_center_y + max_height / 2)
    
    # Ensure the constrained bbox is valid
    if constrained_x_max <= constrained_x_min or constrained_y_max <= constrained_y_min:
        return mask, current_bbox, False
    
    # Check if constraining was actually applied (whether boundaries changed)
    is_constrained = (constrained_x_min > current_bbox[0] or
                      constrained_y_min > current_bbox[1] or
                      constrained_x_max < current_bbox[2] or
                      constrained_y_max < current_bbox[3])
    
    # If no constraining was applied, return original values
    if not is_constrained:
        return mask, current_bbox, False
    
    # Constrain mask
    constrained_mask = mask.copy()
    
    # Clear mask regions outside the constrained area
    # Handle left side of x_min
    if int(constrained_x_min) > 0:
        constrained_mask[:, :int(constrained_x_min)] = False
    
    # Handle above y_min
    if int(constrained_y_min) > 0:
        constrained_mask[:int(constrained_y_min), :] = False
    
    # Handle right side of x_max
    if int(constrained_x_max) < mask.shape[1]:
        constrained_mask[:, int(constrained_x_max):] = False
    
    # Handle below y_max
    if int(constrained_y_max) < mask.shape[0]:
        constrained_mask[int(constrained_y_max):, :] = False
    
    # Recalculate bbox from constrained_mask
    rows = np.any(constrained_mask, axis=1)
    cols = np.any(constrained_mask, axis=0)
    
    # Check if there are valid mask regions
    if np.any(rows) and np.any(cols):
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        constrained_bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
    else:
        # If mask is empty, use constrained coordinates
        constrained_bbox = [int(constrained_x_min), int(constrained_y_min), 
                           int(constrained_x_max), int(constrained_y_max)]
    
    return constrained_mask, constrained_bbox, True

# Clean outlier masks
def clean_outlier_mask(mask, kernel_size=5):
    # Create kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply opening operation (erosion then dilation) to remove small noise
    cleaned_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # Apply closing operation (dilation then erosion) to fill small holes
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    
    return cleaned_mask.astype(bool)

def generate_refined_pklz(args, results_data, exempt_video_ids=None):
    """
    Generate a refined pklz file, updating matched track IDs and optionally adding unmatched segments
    
    Args:
        args: Command line arguments
        results_data: Processing results data
        exempt_video_ids: List of exempt video IDs; these videos will be copied directly from input pklz to output pklz
    """
    print("Generating refined pklz file...")
    
    # If exempt_video_ids is None, initialize as empty list
    if exempt_video_ids is None:
        exempt_video_ids = []
    
    # Create output pklz path
    if args.save_pklz_path:
        output_pklz_path = args.save_pklz_path
    else:
        output_pklz_path = os.path.join(args.output_dir, f'refined_{os.path.basename(args.input_pklz)}')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_pklz_path), exist_ok=True)
    
    # If file already exists, delete it
    if os.path.exists(output_pklz_path):
        os.remove(output_pklz_path)
    
    # Process each video
    with zipfile.ZipFile(args.input_pklz) as input_zf, zipfile.ZipFile(output_pklz_path, 'w') as output_zf:
        # First, get all file names from the input pklz
        input_file_names = input_zf.namelist()
        
        # Find all video IDs
        all_video_ids = set()
        for file_name in input_file_names:
            if file_name.endswith('.pkl') and not file_name.endswith('_image.pkl'):
                video_id = file_name.split('.')[0]
                all_video_ids.add(video_id)
        
        # Process videos that need updating
        for video_id in tqdm(results_data.keys(), desc="Updating pklz file"):
            # Load video data from input pklz
            with input_zf.open(f'{video_id}.pkl') as f:
                preds = pickle.load(f)
            with input_zf.open(f'{video_id}_image.pkl') as f:
                image_preds = pickle.load(f)
            image_ids = sorted(image_preds.id.unique().tolist())
            num_frames = len(image_ids)
            frame_idx_2_image_id = {frame_idx: image_ids[frame_idx] for frame_idx in range(num_frames)}
            
            # Get processing results for the current video
            unprocessed_data_obj = results_data[video_id]['unprocessed_data']
            unmatched_segments = results_data[video_id]['unmatched_segments']
            
            # If fix_duplicate_track_ids is enabled, ensure it's also applied here
            if args.fix_duplicate_track_ids and 'track_ids_fixed' not in results_data[video_id]:
                assert False
            
            # Copy prediction data to avoid modifying the original
            updated_preds = deepcopy(preds)
            updated_image_preds = deepcopy(image_preds)
            
            # Collect indices of unmatched detections to be removed
            unmatched_indices = []
            
            # Process detection results, iterate over all tracklets, segments and detections in unprocessed_data_obj
            for tracklet in unprocessed_data_obj.unprocessed_tracklets:
                for segment in tracklet.segments:
                    for detection in segment.detections:
                        detection_idx = detection.detection_idx
                        original_track_id = detection.track_id
                        
                        if detection.is_matched:
                            # Successfully matched detection, update track_id
                            matched_track_id = detection.matched_track_id
                            
                            mask = updated_preds.index == detection_idx
                            assert mask.any()
                            updated_preds.loc[mask, 'track_id'] = matched_track_id
                        else:
                            # Unmatched detection, add to removal list
                            mask = updated_preds.index == detection_idx
                            assert mask.any()
                            unmatched_indices.append(detection_idx)
            
            # Remove unmatched detections
            if unmatched_indices:
                updated_preds = updated_preds.drop(unmatched_indices)
            
            # If needed, process unmatched segments (add new rows)
            if args.include_unmatched_segments:
                for segment in unmatched_segments:
                    # Handle different segment types (object or dict)
                    frame_idx = segment.frame_idx
                    obj_id = segment.obj_id
                    mask = segment.mask
                    bbox_xyxy = segment.bbox
                    valid_mask = segment.valid_mask
                    
                    # First check if the mask is valid
                    if valid_mask:
                        bbox_ltwh = np.array([bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]])
                        
                        if bbox_ltwh is not None:
                            # Find the image_id corresponding to this frame_idx
                            image_id = frame_idx_2_image_id[frame_idx]
                            
                            # Create new row data
                            # Determine the index for the new row
                            new_idx = updated_preds.index.max() + 1 if not updated_preds.empty else 0
                            
                            # Create new row
                            new_row = pd.DataFrame({
                                'image_id': [image_id],
                                'bbox_ltwh': [bbox_ltwh],
                                'bbox_conf': [1.0],  # Set confidence to 1
                                'video_id': [video_id],
                                'category_id': [1],
                                'track_id': [obj_id],  # track_id becomes a column instead of index level
                                'ignored': [False]
                            }, index=[new_idx])  # Single-level index
                            
                            # Add new row to updated_preds
                            updated_preds = pd.concat([updated_preds, new_row])
            
            # Write updated predictions to output file
            # Create temporary files
            temp_pred_path = f'temp_{video_id}.pkl'
            temp_image_path = f'temp_{video_id}_image.pkl'
            
            # Save updated predictions
            with open(temp_pred_path, 'wb') as f:
                pickle.dump(updated_preds, f)
            
            # Save image predictions (these usually don't need modification)
            with open(temp_image_path, 'wb') as f:
                pickle.dump(updated_image_preds, f)
            
            # Add files to output zip
            output_zf.write(temp_pred_path, f'{video_id}.pkl')
            output_zf.write(temp_image_path, f'{video_id}_image.pkl')
            
            # Remove temporary files
            os.remove(temp_pred_path)
            os.remove(temp_image_path)
            
            # Mark this video ID as processed
            if video_id in all_video_ids:
                all_video_ids.remove(video_id)
        
        # Directly copy exempt videos and all other unprocessed videos
        for video_id in tqdm(list(all_video_ids), desc="Copying unprocessed videos"):
            # Check if this video is in the exempt list
            is_exempt = video_id in exempt_video_ids
            
            if is_exempt:
                print(f"Video {video_id} is in the exempt list, copying directly from input pklz")
            else:
                print(f"Video {video_id} not found in results data, copying directly from input pklz")
            
            # Copy original files directly to output pklz
            with input_zf.open(f'{video_id}.pkl') as f:
                output_zf.writestr(f'{video_id}.pkl', f.read())
            with input_zf.open(f'{video_id}_image.pkl') as f:
                output_zf.writestr(f'{video_id}_image.pkl', f.read())
    
    print(f"Refined pklz file saved to: {output_pklz_path}")
    return output_pklz_path

def fix_duplicate_track_ids_in_data(unprocessed_data_obj, unmatched_segments, sam2_video_data, video_id):
    
    global_obj_id_set = set()
    
    while True:
        has_conflict = False
        frame_to_id = {}

        for tracklet in unprocessed_data_obj.unprocessed_tracklets:
            for segment in tracklet.segments:
                for detection in segment.detections:
                    if detection.is_matched:
                        frame_idx = detection.frame_idx
                        track_id = detection.matched_track_id
                        global_obj_id_set.add(track_id)
                        if frame_idx not in frame_to_id:
                            frame_to_id[frame_idx] = {}
                        if track_id not in frame_to_id[frame_idx]:
                            frame_to_id[frame_idx][track_id] = 0
                        frame_to_id[frame_idx][track_id] += 1
        
        for segment in unmatched_segments:
            frame_idx = segment.frame_idx
            obj_id = segment.obj_id
            global_obj_id_set.add(obj_id)
            if frame_idx not in frame_to_id:
                frame_to_id[frame_idx] = {}
            if obj_id not in frame_to_id[frame_idx]:
                frame_to_id[frame_idx][obj_id] = 0
            frame_to_id[frame_idx][obj_id] += 1
            
        for frame_idx, id_to_cnt in frame_to_id.items():
            for track_id, cnt in id_to_cnt.items():
                if cnt > 1 and not has_conflict:
                    has_conflict = True
                    print(f"Video {video_id}: duplicate track_id {track_id} found at frame_idx {frame_idx}, appearing {cnt} times")
                    
                    segments = sam2_video_data.get_segments_by_obj_id(track_id)
                    segments_at_conflict = []
                    segments_to_change = []

                    for segment in segments:
                        segment_first_index = segment.get_first_frame_idx()
                        segment_last_index = segment.get_last_frame_idx()
                        if segment_first_index <= frame_idx <= segment_last_index:
                            segments_at_conflict.append(segment)
                        elif segment_first_index > frame_idx:
                            segments_to_change.append(segment)
                    
                    segments_at_conflict.sort(key=lambda s: s.get_first_frame_idx())
                    
                    if segments_at_conflict:
                        segments_to_change.extend(segments_at_conflict[1:])
                        
                    if len(segments_to_change) > 0:
                        new_obj_id = track_id + 100
                        while new_obj_id in global_obj_id_set:
                            new_obj_id += 100
                        print(f"Video {video_id}: tracklet {track_id} has conflict, new obj_id is {new_obj_id}, len_segments_to_change: {len(segments_to_change)}")
                            
                        # Update obj_id for all segments that need changing
                        for segment in segments_to_change:
                            # Update the segment's obj_id
                            segment.obj_id = new_obj_id
                            
                            # Update obj_id for all detections in this segment
                            for frame_idx, detection in segment.get_all_detections().items():
                                detection.obj_id = new_obj_id
                                
                                # Update matched_track_id of the matched unprocessed_detection
                                if detection.matched_unprocessed_detection:
                                    detection.matched_unprocessed_detection.matched_track_id = new_obj_id
                                
                                # Update obj_id of the associated unmatched_segment
                                if detection.unmatched_segment:
                                    detection.unmatched_segment.obj_id = new_obj_id
                    else:
                        print(f"Video {video_id}: tracklet {track_id} has conflict but no segments need changing")
                        duplicated_unprocessed_detections = []
                        for segment in segments:
                            for frame_idx2, detection in segment.get_all_detections().items():
                                if detection.is_matched:
                                    if frame_idx2 == frame_idx and detection.matched_track_id == track_id:
                                        duplicated_unprocessed_detections.append(detection)
                        duplicated_unmatched_segments = []
                        for segment in unmatched_segments:
                            if segment.frame_idx == frame_idx and segment.obj_id == track_id:
                                duplicated_unmatched_segments.append(segment)
                        
                        duplicated_ones = duplicated_unprocessed_detections + duplicated_unmatched_segments
                        # Except the first one, assign a new ID to each of them
                        if len(duplicated_ones) > 1:
                            # Keep the first, assign new IDs to the rest
                            for i in range(1, len(duplicated_ones)):
                                new_obj_id = track_id + 100
                                while new_obj_id in global_obj_id_set:
                                    new_obj_id += 100
                                
                                print(f"Video {video_id}: duplicate track_id {track_id} at frame {frame_idx}, assigning new ID {new_obj_id} to the {i+1}th duplicate")
                                
                                # Update ID based on type
                                if isinstance(duplicated_ones[i], unprocessed_detection):
                                    duplicated_ones[i].matched_track_id = new_obj_id
                                    # If there's an associated sam2_detection, update it too
                                    if duplicated_ones[i].matched_sam2_detection:
                                        duplicated_ones[i].matched_sam2_detection.obj_id = new_obj_id
                                elif hasattr(duplicated_ones[i], 'obj_id'):  # unmatched_segment
                                    duplicated_ones[i].obj_id = new_obj_id
                                
                                # Add new ID to global set
                                global_obj_id_set.add(new_obj_id)
                            
                            has_conflict = True
                            modified_cnt += 1
                        
                                
        if not has_conflict:
            break
    

class unprocessed_detection:
    def __init__(self, detection_idx, frame_idx, bbox_ltwh, bbox_xyxy, track_id, is_matched, matched_track_id, tried_propagation, segment_length, is_short_segment):
        self.detection_idx = detection_idx
        self.frame_idx = frame_idx
        self.bbox_ltwh = bbox_ltwh
        self.bbox_xyxy = bbox_xyxy
        self.track_id = track_id
        self.is_matched = is_matched
        self.matched_track_id = matched_track_id
        self.tried_propagation = tried_propagation
        self.segment_length = segment_length
        self.is_short_segment = is_short_segment
        self.mask = None
        # New fields for bidirectional linking
        self.parent_segment = None  # Reference to the parent segment
        self.matched_sam2_detection = None  # Reference to the matched sam2_detection
        
    def __repr__(self):
        return f"unprocessed_detection(frame_idx={self.frame_idx}, track_id={self.track_id}, is_matched={self.is_matched}, bbox={self.bbox_xyxy})"
        
    def get_parent_segment(self):
        """Return the parent unprocessed_segment this detection belongs to"""
        return self.parent_segment
        
    def get_parent_tracklet(self):
        """Return the parent unprocessed_tracklet this detection belongs to"""
        if self.parent_segment:
            return self.parent_segment.parent_tracklet
        return None
        
# Consecutive unprocessed_detections with the same track_id
class unprocessed_segment:
    def __init__(self, track_id):
        self.track_id = track_id
        self.detections = []
        self.parent_tracklet = None  # Reference to the parent tracklet
        
    def __repr__(self):
        num_detections = len(self.detections)
        frame_range = f"{self.detections[0].frame_idx}-{self.detections[-1].frame_idx}" if num_detections > 0 else "empty"
        return f"unprocessed_segment(track_id={self.track_id}, detections={num_detections}, frames={frame_range})"
        
    def add_detection(self, detection: unprocessed_detection):
        self.detections.append(detection)
        detection.parent_segment = self  # Set this segment as the parent of the detection
        
# Consecutive unprocessed_segments with the same track_id
class unprocessed_tracklet:
    def __init__(self, track_id):
        self.track_id = track_id
        self.segments = []
        
    def __repr__(self):
        num_segments = len(self.segments)
        total_detections = sum(len(segment.detections) for segment in self.segments)
        return f"unprocessed_tracklet(track_id={self.track_id}, segments={num_segments}, total_detections={total_detections})"
        
    def add_segment(self, segment: unprocessed_segment):
        self.segments.append(segment)
        segment.parent_tracklet = self  # Set this tracklet as the parent of the segment
        
class unprocessed_data:
    def __init__(self, unprocessed_tracklets: list[unprocessed_tracklet]):
        self.unprocessed_tracklets = unprocessed_tracklets
        # Create a dict to index tracklets by track_id
        self.track_id_to_tracklet = {}
        for tracklet in unprocessed_tracklets:
            self.track_id_to_tracklet[tracklet.track_id] = tracklet
    
    def __repr__(self):
        num_tracklets = len(self.unprocessed_tracklets)
        total_segments = sum(len(tracklet.segments) for tracklet in self.unprocessed_tracklets)
        unique_track_ids = len(self.track_id_to_tracklet)
        return f"unprocessed_data(tracklets={num_tracklets}, unique_track_ids={unique_track_ids}, total_segments={total_segments})"
    
    def get_tracklet_by_id(self, track_id):
        return self.track_id_to_tracklet.get(track_id, None)
        
class sam2_detection:
    def __init__(self, frame_idx, obj_id, mask, valid_mask, has_outlier=False, bbox=None, is_constrained=False):
        self.frame_idx = frame_idx
        self.obj_id = obj_id
        self.mask = mask
        self.valid_mask = valid_mask
        self.has_outlier = has_outlier
        self.bbox = bbox
        self.is_constrained = is_constrained
        # New fields for bidirectional linking
        self.parent_segment = None  # Reference to the parent sam2_video_segment
        self.matched_unprocessed_detection = None  # Reference to the matched unprocessed_detection
        self.unmatched_segment = None  # Reference to the derived unmatched_segment if any
        
    def __repr__(self):
        mask_shape = self.mask.shape if self.mask is not None else None
        return f"sam2_detection(frame={self.frame_idx}, obj_id={self.obj_id}, valid={self.valid_mask}, bbox={self.bbox}, mask_shape={mask_shape})"
        
    def get_parent_segment(self):
        """Return the parent sam2_video_segment this detection belongs to"""
        return self.parent_segment
        
    def get_unmatched_segment(self):
        """Return the unmatched_segment derived from this detection, if any"""
        return self.unmatched_segment
        
    def set_unmatched_segment(self, segment):
        """Link this detection to an unmatched_segment"""
        self.unmatched_segment = segment

class sam2_video_segment:
    def __init__(self, obj_id):
        self.obj_id = obj_id
        self.detections = {}
        self.first_frame_idx = None
        self.last_frame_idx = None
        self.parent_video_data = None
        self.parent_tracklet = None  # Reference to the parent tracklet
        
    def __repr__(self):
        num_detections = len(self.detections)
        frame_range = f"{self.first_frame_idx}-{self.last_frame_idx}" if self.first_frame_idx is not None else "empty"
        return f"sam2_video_segment(obj_id={self.obj_id}, detections={num_detections}, frames={frame_range})"
        
    def add_sam2_detection(self, sam2_detection: sam2_detection):
        self.detections[sam2_detection.frame_idx] = sam2_detection
        sam2_detection.parent_segment = self  # Set this segment as the parent of the detection
        if self.first_frame_idx is None or sam2_detection.frame_idx < self.first_frame_idx:
            self.first_frame_idx = sam2_detection.frame_idx
        if self.last_frame_idx is None or sam2_detection.frame_idx > self.last_frame_idx:
            self.last_frame_idx = sam2_detection.frame_idx
        
    def get_sam2_detection(self, frame_idx):
        return self.detections.get(frame_idx, None)
    
    def get_all_detections(self):
        return self.detections
    
    def get_frame_indices(self):
        return sorted(list(self.detections.keys()))
        
    def get_first_frame_idx(self):
        return self.first_frame_idx
    
    def get_last_frame_idx(self):
        return self.last_frame_idx
    
# Stores all valid segments for the entire video, indexable by obj_id
class sam2_video_data:
    def __init__(self):
        self.segments = []
        self.tracklets = []  # Stores all tracklets
        
    def __repr__(self):
        num_segments = len(self.segments)
        num_tracklets = len(self.tracklets)
        unique_obj_ids = len(set(segment.obj_id for segment in self.segments))
        return f"sam2_video_data(segments={num_segments}, tracklets={num_tracklets}, unique_obj_ids={unique_obj_ids})"
        
    def add_segment(self, sam2_video_segment: sam2_video_segment):
        self.segments.append(sam2_video_segment)
        sam2_video_segment.parent_video_data = self  # Set this video_data as the parent of the segment
        
    def get_segments_by_obj_id(self, obj_id):
        return [segment for segment in self.segments if segment.obj_id == obj_id]
    
    def get_all_detections_in_frame(self, frame_idx):
        detections = []
        for segment in self.segments:
            detection = segment.get_sam2_detection(frame_idx)
            if detection:
                detections.append(detection)
        return detections
    
    def create_tracklets(self):
        """
        Create tracklets by grouping segments with the same obj_id
        """
        # Clear existing tracklets
        self.tracklets = []
        
        # Group segments by obj_id
        obj_ids = set(segment.obj_id for segment in self.segments)
        
        # Create a tracklet for each obj_id
        for obj_id in obj_ids:
            segments = self.get_segments_by_obj_id(obj_id)
            if segments:
                tracklet = sam2_video_tracklet(obj_id)
                for segment in segments:
                    tracklet.add_segment(segment)
                self.tracklets.append(tracklet)
                
        return self.tracklets
    
    def get_tracklet_by_obj_id(self, obj_id):
        """
        Get the tracklet for a specified obj_id
        """
        # If tracklets are empty, create them first
        if not self.tracklets:
            self.create_tracklets()
            
        # Find the corresponding tracklet
        for tracklet in self.tracklets:
            if tracklet.obj_id == obj_id:
                return tracklet
        return None

class sam2_video_tracklet:
    def __init__(self, obj_id):
        self.obj_id = obj_id
        self.segments = []
        self.parent_video_data = None
        
    def __repr__(self):
        num_segments = len(self.segments)
        total_detections = sum(len(segment.detections) for segment in self.segments)
        frame_range = self.get_frame_range()
        frame_str = f"{frame_range[0]}-{frame_range[1]}" if frame_range[0] is not None else "empty"
        return f"sam2_video_tracklet(obj_id={self.obj_id}, segments={num_segments}, detections={total_detections}, frames={frame_str})"
        
    def add_segment(self, segment: sam2_video_segment):
        """
        Add a segment to the current tracklet
        """
        self.segments.append(segment)
        segment.parent_tracklet = self  # Set the segment's parent_tracklet reference
        
    def get_all_segments(self):
        """
        Get all segments of the current tracklet
        """
        return self.segments
    
    def get_all_detections(self):
        """
        Get all detections of the current tracklet
        """
        all_detections = []
        for segment in self.segments:
            all_detections.extend(segment.get_all_detections().values())
        return all_detections
    
    def get_all_detections_in_frame(self, frame_idx):
        """
        Get all detections of the current tracklet at a specified frame
        """
        detections = []
        for segment in self.segments:
            detection = segment.get_all_detections_in_frame(frame_idx)
            if detection:
                detections.append(detection)
        return detections
    
    def get_frame_range(self):
        """
        Get the frame range of the current tracklet
        """
        all_detections = self.get_all_detections()
        if not all_detections:
            return None, None
        
        frame_indices = [detection.frame_idx for detection in all_detections]
        return min(frame_indices), max(frame_indices)

class unmatched_segment:
    def __init__(self, frame_idx, obj_id, mask, valid_mask, bbox=None, is_constrained=False, has_outlier=False):
        self.frame_idx = frame_idx
        self.obj_id = obj_id
        self.mask = mask
        self.valid_mask = valid_mask
        self.bbox = bbox
        self.is_constrained = is_constrained
        self.has_outlier = has_outlier
        # Bidirectional linking fields
        self.source_sam2_detection = None  # Associated sam2_detection
    
    def __repr__(self):
        mask_shape = self.mask.shape if self.mask is not None else None
        return f"unmatched_segment(frame={self.frame_idx}, obj_id={self.obj_id}, valid={self.valid_mask}, bbox={self.bbox}, mask_shape={mask_shape})"
    
    def get_source_sam2_detection(self):
        """Get the sam2_detection that generated this unmatched_segment"""
        return self.source_sam2_detection
    
    def get_source_segment(self):
        """Get the sam2_video_segment that generated this unmatched_segment"""
        if self.source_sam2_detection:
            return self.source_sam2_detection.get_parent_segment()
        return None
    
    @classmethod
    def from_dict(cls, segment_dict):
        """Create an instance from a dictionary, compatible with legacy code"""
        segment = cls(
            frame_idx=segment_dict['frame_idx'],
            obj_id=segment_dict['obj_id'],
            mask=segment_dict['segment_info']['mask'],
            valid_mask=segment_dict['segment_info']['valid_mask'],
            bbox=segment_dict['segment_info']['bbox'],
            is_constrained=segment_dict['segment_info'].get('is_constrained', False),
            has_outlier=segment_dict['segment_info'].get('has_outlier', False)
        )
        return segment
    
    @classmethod
    def from_sam2_detection(cls, sam2_detection):
        """Create an instance from a sam2_detection"""
        segment = cls(
            frame_idx=sam2_detection.frame_idx,
            obj_id=sam2_detection.obj_id,
            mask=deepcopy(sam2_detection.mask) if sam2_detection.mask is not None else None,
            valid_mask=sam2_detection.valid_mask,
            bbox=sam2_detection.bbox,
            is_constrained=sam2_detection.is_constrained,
            has_outlier=sam2_detection.has_outlier
        )
        segment.source_sam2_detection = sam2_detection
        return segment

def break_circular_references(data_objects):
    """
    Break circular references in data structures for serialization and storage
    
    Args:
        data_objects: Dict containing data objects to process (unprocessed_data_obj, unmatched_segments, video_data)
    
    Returns:
        Processed copy of data objects that can be safely serialized
    """
    print("Removing circular references...")
    
    # Create deep copy to avoid modifying original data
    result = {}
    
    for key, obj in data_objects.items():
        # Handle different object types
        if key == 'unprocessed_data':
            unprocessed_data_obj = obj
            # Remove circular references from unprocessed_data
            for tracklet in unprocessed_data_obj.unprocessed_tracklets:
                for segment in tracklet.segments:
                    # Remove segment's reference to tracklet
                    segment.parent_tracklet = None
                    
                    for detection in segment.detections:
                        # Remove detection's reference to segment
                        detection.parent_segment = None
                        
                        # Remove bidirectional link with sam2_detection
                        detection.matched_sam2_detection = None
            
            result[key] = unprocessed_data_obj
            
        elif key == 'unmatched_segments':
            unmatched_segments = obj
            # Remove circular references from unmatched_segments
            for segment in unmatched_segments:
                # Remove reference to source_sam2_detection
                segment.source_sam2_detection = None
            
            result[key] = unmatched_segments
            
        else:
            # Other object types are added directly
            result[key] = obj
    
    return result

# Calculate mask connected components
def calculate_mask_connected_components(mask):
    """
    Calculate the number of connected components in a binary mask
    
    Args:
        mask: Binary mask array (H,W)
        
    Returns:
        num_components: Number of connected components
    """
    # Ensure mask is uint8 type
    mask_uint8 = mask.astype(np.uint8)
    
    # Use OpenCV to calculate connected components
    num_components, _ = cv2.connectedComponents(mask_uint8, connectivity=8)
    
    # Subtract background component, return actual number of connected components
    return num_components - 1
