import os
import json
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
import math
from math import floor
from torch.utils.data import DataLoader
from utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xywh, bbox_xywh_to_cxcywh
from data.utils import Compose, ToTensor, RandomResize, Normalize, get_image_hw, ColorJitter, RandomHorizontalFlip, GaussianNoise, GaussianBlur, ClearAugmentationMetas, RandomCrop, RandomAffine, RandomPerspective
from data.pnlcalib_utils.utils_keypoints import KeypointsDB
from data.pnlcalib_utils.utils_lines import LineKeypointsDB
import copy
import zipfile
import pickle

from data.utils import flip_annot_names, h_lines, v_lines, correct_lines_labels, get_visible_lines_coords

role_mapping = {'ball': 0, 'goalkeeper': 1, 'other': 2, 'player': 3, 'referee': 4, None: 5}
reid_columns = ["role", "team", "filtered_jersey_number", "digit_head", "digit_tail"]
jn_mapping = {str(i): i for i in range(100)}
jn_mapping[None] = 100
digit_head_mapping = {str(i): i-1 for i in range(1, 10)}
digit_head_mapping[None] = 9
digit_tail_mapping = {str(i): i for i in range(10)}
digit_tail_mapping[None] = 10

class SoccerNetGSR_Detection(Dataset):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "",
            split: str = "train",
            transforms=None,
            num_keypoints: int = 58,
            num_lines: int = 24,
            detection_data_type: str = "image",
            backbone_type: str = "image",
            num_frames: int = 30,
            image_input_size: int = 512,
            detect_ball: bool = False,
            detect_ball_only: bool = False,
            use_soccer_factory_data: bool = False,
            soccer_factory_data_path: str = "",
            soccer_factory_data_only: bool = False,
            use_soccer_factory_data_amount: int = -1,
            train_keypoints_or_lines_detection: bool = True,
    ):
        super(SoccerNetGSR_Detection, self).__init__()
        assert split in ['train', 'valid', 'test']
        
        self.data_dir = os.path.join(data_root, sub_dir)
        self.split = split
        self.transforms = transforms
        self.num_keypoints = num_keypoints
        self.num_lines = num_lines
        self.detection_data_type = detection_data_type
        self.backbone_type = backbone_type
        self.num_frames = num_frames
        self.image_input_size = image_input_size
        self.detect_ball = detect_ball
        self.detect_ball_only = detect_ball_only
        self.use_soccer_factory_data = use_soccer_factory_data
        self.soccer_factory_data_path = soccer_factory_data_path
        self.soccer_factory_data_only = soccer_factory_data_only
        self.use_soccer_factory_data_amount = use_soccer_factory_data_amount
        self.train_keypoints_or_lines_detection = train_keypoints_or_lines_detection
        
        # Validate configuration
        if self.detect_ball_only and self.detect_ball:
            print("Warning: Both detect_ball_only and detect_ball are set to True. detect_ball_only takes precedence.")

        self.sequence_infos = self._get_sequence_infos()
        self.image_paths = defaultdict(list)
        
        self.annotations = dict()
        
        self.soccer_factory_data_sequences = set()
        self.soccer_factory_data_pkl_paths = dict()
        
        if not (soccer_factory_data_only and self.split == 'train'):
            image_paths = self._get_image_paths()
            self.image_paths.update(image_paths)
            annotations = self._get_annotations()
            self.annotations.update(annotations)
        if use_soccer_factory_data and self.split == 'train':
            self._init_soccer_factory_data_lazy()
            
        self.set_sample_position()
            
        return

    def get_sequence_infos(self):
        return self.sequence_infos

    def get_image_paths(self):
        return self.image_paths

    def _get_sequence_names(self):
        sequence_names = os.listdir(os.path.join(self.data_dir, 'SoccerNetGS', self.split))
        return [name for name in sequence_names if os.path.isdir(os.path.join(self.data_dir, 'SoccerNetGS', self.split, name))]

    def _get_sequence_infos(self):
        sequence_names = self._get_sequence_names()
        sequence_infos = dict()
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            metadata_path = os.path.join(sequence_dir, "Labels-GameState.json")
            metadata = json.load(open(metadata_path))
            sequence_infos[sequence_name] = {
                "width": 1920,
                "height": 1080,
                "length": int(metadata['info']['seq_length']),
                "is_static": False,
                "is_soccer_factory_data": False,
            }
        return sequence_infos

    def _get_image_paths(self):
        sequence_names = self._get_sequence_names()
        image_paths = defaultdict(list)
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            for i in range(self.sequence_infos[sequence_name]["length"]):
                image_paths[sequence_name].append(self._get_image_path(sequence_dir, i))
        return image_paths

    @staticmethod
    def _get_sequence_dir(data_dir, split, sequence_name):
        return str(os.path.join(data_dir, 'SoccerNetGS', split, sequence_name))

    @staticmethod
    def _get_image_path(sequence_dir, frame_idx):
        return str(os.path.join(sequence_dir, "img1", f"{frame_idx+1:06d}.jpg"))    # the image name is 1-indexed
            
    def _init_annotations(self, sequence_names, soccer_factory_data=False):
        annotations = dict()
        for sequence_name in sequence_names:
            annotations[sequence_name] = []
            num_frames = self.sequence_infos[sequence_name]["length"]

            for i in range(num_frames):
                annotations[sequence_name].append({
                    "id": [],
                    "category": [],
                    "bbox": [],
                    "visibility": [],
                    "role": [],
                    "jersey": [],
                    "digit_head": [],
                    "digit_tail": [],
                    "legibility_score": [],
                    "lines": {},
                })
        return annotations
    
    def _get_annotations(self):
        
        # get legibility jn info
        legibility_jn_json_path = os.path.join(self.data_dir, 'legibility_jn', f'{self.split}.json')
        with open(legibility_jn_json_path, 'r') as f:
            legibility_jn_info = json.load(f)
        legibility_jn_dict = {}
        for [sequence_id, image_id, track_id, jn, legibility] in legibility_jn_info:
            legibility_jn_dict.update({(sequence_id, image_id, track_id): (jn, legibility)})
        
        sequence_names = self._get_sequence_names()
        # Init the annotations:
        annotations = self._init_annotations(sequence_names)
        # Load the annotations:
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            gt_file_path = os.path.join(sequence_dir, "Labels-GameState.json")
            gt = json.load(open(gt_file_path))
            annos = gt['annotations']
            for anno in annos:
                # Filter based on detect_ball and detect_ball_only parameters
                if self.detect_ball_only:
                    # Only include ball (exclude person)
                    if not ((anno['supercategory'] == 'object' and anno['attributes']['role'] == 'ball') or (anno['supercategory']== 'pitch')):
                        continue
                elif self.detect_ball:
                    # Include both person and ball
                    if not ((anno['supercategory'] == 'object') or (anno['supercategory']== 'pitch')):
                        continue
                else:
                    # Only include person (exclude ball)
                    if not ((anno['supercategory'] == 'object' and anno['attributes']['role'] != 'ball') or (anno['supercategory']== 'pitch')):
                        continue
                
                frame_idx = int(anno['image_id'][-6:]) - 1
                if anno['supercategory'] == 'object':
                    obj_id = anno['track_id']
                    x, y, w, h = anno['bbox_image']['x'], anno['bbox_image']['y'], anno['bbox_image']['w'], anno['bbox_image']['h']
                    bbox = [x, y, w, h]
                    visibility = 1.0
                    
                    # Set category based on detection mode
                    if self.detect_ball_only:
                        # Only ball detection: ball -> 0
                        if anno['attributes']['role'] == 'ball':
                            category = 0
                        else:
                            # This should not happen due to filtering, but handle it gracefully
                            continue
                    else:
                        # Normal or ball+person detection: person -> 0, ball -> 1
                        if anno['attributes']['role'] == 'ball':
                            category = 1
                        else:
                            category = 0
                    
                    # Append to lists instead of using torch.cat
                    annotations[sequence_name][frame_idx]["id"].append(obj_id)
                    annotations[sequence_name][frame_idx]["category"].append(category)
                    annotations[sequence_name][frame_idx]["bbox"].append(bbox)
                    annotations[sequence_name][frame_idx]["visibility"].append(visibility)
                    
                    # For ball, set all attributes to default values
                    if anno['attributes']['role'] == 'ball':
                        annotations[sequence_name][frame_idx]["role"].append(role_mapping['ball'])
                        annotations[sequence_name][frame_idx]["legibility_score"].append(0.0)
                        annotations[sequence_name][frame_idx]["jersey"].append(jn_mapping[None])
                        annotations[sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[None])
                        annotations[sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[None])
                    else:
                        # For person, process attributes as before
                        annotations[sequence_name][frame_idx]["role"].append(role_mapping[anno['attributes']['role']])
                        
                        # get legibility score $$ filtered jn
                        sequence_id = sequence_name[-3:]
                        image_id = anno['image_id']
                        track_id = anno['track_id']
                        legibility_score = legibility_jn_dict[(sequence_id, image_id, track_id)][1]
                        annotations[sequence_name][frame_idx]["legibility_score"].append(legibility_score)
                        jn = anno['attributes']['jersey'] if legibility_score > 0.5 else None
                        annotations[sequence_name][frame_idx]["jersey"].append(jn_mapping[jn])
                        # get digit head and digit tail
                        if jn is not None:
                            if len(jn) == 1:
                                annotations[sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[jn])
                                annotations[sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[None])
                            elif len(jn) == 2:
                                annotations[sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[jn[0]])
                                annotations[sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[jn[1]])
                            else:
                                annotations[sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[None])
                                annotations[sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[None])
                        else:
                            annotations[sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[None])
                            annotations[sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[None])
                elif anno['supercategory']== 'pitch':
                    annotations[sequence_name][frame_idx]['lines'] = correct_lines_labels(anno['lines'])
                    annotations[sequence_name][frame_idx]['valid_lines'] = True
                    annotations[sequence_name][frame_idx]['valid_keypoints'] = True
                else:
                    raise ValueError(f"Unknown annotation: {anno}")
                
        # Convert lists to tensors in a single operation per frame
        for sequence_name in sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                frame_annotation = annotations[sequence_name][i]
                if len(frame_annotation["id"]) > 0:
                    frame_annotation["id"] = torch.tensor(frame_annotation["id"], dtype=torch.int64)
                    frame_annotation["category"] = torch.tensor(frame_annotation["category"], dtype=torch.int64)
                    frame_annotation["bbox"] = torch.tensor(frame_annotation["bbox"], dtype=torch.float32)
                    frame_annotation["visibility"] = torch.tensor(frame_annotation["visibility"], dtype=torch.float32)
                    frame_annotation["role"] = torch.tensor(frame_annotation["role"], dtype=torch.int64)
                    frame_annotation["jersey"] = torch.tensor(frame_annotation["jersey"], dtype=torch.int64)
                    frame_annotation["digit_head"] = torch.tensor(frame_annotation["digit_head"], dtype=torch.int64)
                    frame_annotation["digit_tail"] = torch.tensor(frame_annotation["digit_tail"], dtype=torch.int64)
                    frame_annotation["legibility_score"] = torch.tensor(frame_annotation["legibility_score"], dtype=torch.float32)
                else:
                    # Empty frame
                    frame_annotation["id"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["category"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["bbox"] = torch.zeros((0, 4), dtype=torch.float32)
                    frame_annotation["visibility"] = torch.zeros((0, ), dtype=torch.float32)
                    frame_annotation["role"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["jersey"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["digit_head"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["digit_tail"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["legibility_score"] = torch.zeros((0, ), dtype=torch.float32)

        # Determine whether each annotation is legal:
        for sequence_name in sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name][i]["is_legal"] = is_legal(annotations[sequence_name][i])
        return annotations
    
    def _init_soccer_factory_data_lazy(self):
        """
        Initialize extra data lazily: only load metadata and paths, not actual annotations.
        """
        soccer_factory_data_dir = os.path.dirname(self.soccer_factory_data_path)
        if self.soccer_factory_data_path.endswith('.pkl'):
            soccer_factory_data_dir = self.soccer_factory_data_path.replace('.pkl', '')
        
        # check directory exists
        if not os.path.exists(soccer_factory_data_dir):
            print(f"Warning: Extra data directory not found: {soccer_factory_data_dir}")
            print("Please run split_extracted_info.py first to split the pkl file.")
            return
        
        pkl_files = [f for f in os.listdir(soccer_factory_data_dir) if f.endswith('.pkl')]
        
        if self.use_soccer_factory_data_amount >= 0:
            pkl_files_with_idx = [(f, int(f.split('-')[-1].replace('.pkl', '')[-5:])) for f in pkl_files]
            pkl_files_with_idx.sort(key=lambda x: x[1])
            pkl_files = [f for f, idx in pkl_files_with_idx[:self.use_soccer_factory_data_amount]]
        
        print(f"Found {len(pkl_files)} extra data sequences to load lazily")
        
        for pkl_file in pkl_files:
            processed_sequence_name = pkl_file.replace('.pkl', '')
            pkl_path = os.path.join(soccer_factory_data_dir, pkl_file)
            
            with open(pkl_path, 'rb') as f:
                sequence_data = pickle.load(f)
            num_frames = len(sequence_data)
            del sequence_data
            
            self.sequence_infos[processed_sequence_name] = {
                "width": 1920,
                "height": 1080,
                "length": num_frames,
                "is_static": False,
                "is_soccer_factory_data": True,
            }
            
            self.soccer_factory_data_sequences.add(processed_sequence_name)
            self.soccer_factory_data_pkl_paths[processed_sequence_name] = pkl_path
            
            sequence_dir = self._get_sequence_dir(self.data_dir, 'sn500', processed_sequence_name)
            for i in range(num_frames):
                image_path = self._get_image_path(sequence_dir, i)
                self.image_paths[processed_sequence_name].append(image_path)
            
            self.annotations[processed_sequence_name] = []
            for i in range(num_frames):
                self.annotations[processed_sequence_name].append({
                    "is_legal": True,
                    "lazy_load": True,
                })
        
        print(f"Lazy load initialization completed for {len(pkl_files)} sequences")
    
    def _process_soccer_factory_data_frame(self, sequence_name, frame_id, frame_data):
        """
        Process a single frame's data and convert it to an annotation dict.

        Args:
            sequence_name: e.g. 'SNGS-0001'.
            frame_id: Frame ID (1-indexed).
            frame_data: Raw data for the frame.

        Returns:
            Annotation dict for the frame.
        """
        frame_annotation = {
            "id": [],
            "category": [],
            "bbox": [],
            "visibility": [],
            "role": [],
            "jersey": [],
            "digit_head": [],
            "digit_tail": [],
            "legibility_score": [],
        }
        
        if 'people' in frame_data:
            for person in frame_data['people']:
                frame_annotation["id"].append(person['id'])
                frame_annotation["category"].append(0)
                frame_annotation["bbox"].append(person['bbox_ltwh'].tolist())
                frame_annotation["visibility"].append(1.0)
                frame_annotation["role"].append(role_mapping[person['role']])
                frame_annotation["legibility_score"].append(person['legibility_score'])
                
                jn = person['jersey_number'] if person['legibility_score'] > 0.5 else None
                jn = str(int(jn)) if jn is not None else None
                if (jn is not None) and (int(jn) < 0 or int(jn) > 99):
                    jn = None
                
                frame_annotation["jersey"].append(jn_mapping[jn])
                if jn is not None:
                    if len(jn) == 1:
                        frame_annotation["digit_tail"].append(digit_tail_mapping[jn])
                        frame_annotation["digit_head"].append(digit_head_mapping[None])
                    elif len(jn) == 2:
                        frame_annotation["digit_head"].append(digit_head_mapping[jn[0]])
                        frame_annotation["digit_tail"].append(digit_tail_mapping[jn[1]])
                    else:
                        frame_annotation["digit_head"].append(digit_head_mapping[None])
                        frame_annotation["digit_tail"].append(digit_tail_mapping[None])
                else:
                    frame_annotation["digit_head"].append(digit_head_mapping[None])
                    frame_annotation["digit_tail"].append(digit_tail_mapping[None])
        
        if frame_data['valid_cam_params']:
            frame_annotation["valid_lines"] = True
            frame_annotation["valid_keypoints"] = True
            frame_annotation["lines"] = correct_lines_labels(get_visible_lines_coords(
                frame_data["K"], frame_data["R"], 
                self.sequence_infos[sequence_name]["height"], 
                self.sequence_infos[sequence_name]["width"]))
        else:
            frame_annotation["valid_lines"] = False
            frame_annotation["valid_keypoints"] = False
            frame_annotation["lines"] = {}
        
        if len(frame_annotation["id"]) > 0:
            frame_annotation["id"] = torch.tensor(frame_annotation["id"], dtype=torch.int64)
            frame_annotation["category"] = torch.tensor(frame_annotation["category"], dtype=torch.int64)
            frame_annotation["bbox"] = torch.tensor(frame_annotation["bbox"], dtype=torch.float32)
            frame_annotation["visibility"] = torch.tensor(frame_annotation["visibility"], dtype=torch.float32)
            frame_annotation["role"] = torch.tensor(frame_annotation["role"], dtype=torch.int64)
            frame_annotation["jersey"] = torch.tensor(frame_annotation["jersey"], dtype=torch.int64)
            frame_annotation["digit_head"] = torch.tensor(frame_annotation["digit_head"], dtype=torch.int64)
            frame_annotation["digit_tail"] = torch.tensor(frame_annotation["digit_tail"], dtype=torch.int64)
            frame_annotation["legibility_score"] = torch.tensor(frame_annotation["legibility_score"], dtype=torch.float32)
        else:
            # Empty frame
            frame_annotation["id"] = torch.zeros((0, ), dtype=torch.int64)
            frame_annotation["category"] = torch.zeros((0, ), dtype=torch.int64)
            frame_annotation["bbox"] = torch.zeros((0, 4), dtype=torch.float32)
            frame_annotation["visibility"] = torch.zeros((0, ), dtype=torch.float32)
            frame_annotation["role"] = torch.zeros((0, ), dtype=torch.int64)
            frame_annotation["jersey"] = torch.zeros((0, ), dtype=torch.int64)
            frame_annotation["digit_head"] = torch.zeros((0, ), dtype=torch.int64)
            frame_annotation["digit_tail"] = torch.zeros((0, ), dtype=torch.int64)
            frame_annotation["legibility_score"] = torch.zeros((0, ), dtype=torch.float32)
        
        frame_annotation["is_legal"] = is_legal(frame_annotation)
        
        return frame_annotation
    
    def _load_soccer_factory_data_frames(self, sequence_name, frame_indices):
        """
        Load annotations for multiple frames from disk, reading the pkl file once.

        Args:
            sequence_name: e.g. 'SNGS-0001'.
            frame_indices: List of frame indices (0-indexed).

        Returns:
            List of frame annotation dicts.
        """
        pkl_path = self.soccer_factory_data_pkl_paths[sequence_name]
        with open(pkl_path, 'rb') as f:
            sequence_data = pickle.load(f)
        
        annotations = []
        for frame_idx in frame_indices:
            frame_id = frame_idx + 1
            
            if frame_id in sequence_data:
                frame_data = sequence_data[frame_id]
                frame_annotation = self._process_soccer_factory_data_frame(sequence_name, frame_id, frame_data)
            else:
                frame_annotation = {
                    "id": torch.zeros((0, ), dtype=torch.int64),
                    "category": torch.zeros((0, ), dtype=torch.int64),
                    "bbox": torch.zeros((0, 4), dtype=torch.float32),
                    "visibility": torch.zeros((0, ), dtype=torch.float32),
                    "role": torch.zeros((0, ), dtype=torch.int64),
                    "jersey": torch.zeros((0, ), dtype=torch.int64),
                    "digit_head": torch.zeros((0, ), dtype=torch.int64),
                    "digit_tail": torch.zeros((0, ), dtype=torch.int64),
                    "legibility_score": torch.zeros((0, ), dtype=torch.float32),
                    "valid_lines": False,
                    "valid_keypoints": False,
                    "lines": {},
                    "is_legal": True,
                }
            
            annotations.append(frame_annotation)
        
        del sequence_data
        
        return annotations
    
    def _get_soccer_factory_data_image_paths(self, soccer_factory_data):
        """
        Get image paths for extra data sequences
        """
        image_paths = defaultdict(list)
        
        sequence_names = list(set(soccer_factory_data.keys()))
        
        processed_sequence_names = [f'SNGS-{name}' for name in sequence_names]
        
        for name, processed_sequence_name in zip(sequence_names, processed_sequence_names):
            sequence_dir = self._get_sequence_dir(self.data_dir, 'sn500', processed_sequence_name)
            
            num_frames = len(soccer_factory_data[name])
            self.sequence_infos[processed_sequence_name] = {
                "width": 1920,
                "height": 1080,
                "length": num_frames,
                "is_static": False,
                "is_soccer_factory_data": True,
            }
            
            sequence_length = self.sequence_infos[processed_sequence_name]["length"]
            
            for i in range(sequence_length):
                image_path = self._get_image_path(sequence_dir, i)
                image_paths[processed_sequence_name].append(image_path)
        
        return image_paths
    
    def _get_soccer_factory_data_annotations(self, soccer_factory_data):
        
        processed_sequence_names = [f'SNGS-{vid}' for vid in soccer_factory_data.keys()]
        annotations = self._init_annotations(processed_sequence_names, soccer_factory_data=True)
        
        for vid in soccer_factory_data.keys():
            processed_sequence_name = f'SNGS-{vid}'
            sequence_length = self.sequence_infos[processed_sequence_name]["length"]
            for frame_id in soccer_factory_data[vid].keys():
                frame_idx = frame_id - 1

                if 'people' in soccer_factory_data[vid][frame_id].keys():
                    for person in soccer_factory_data[vid][frame_id]['people']:
                        annotations[processed_sequence_name][frame_idx]["id"].append(person['id'])
                        annotations[processed_sequence_name][frame_idx]["category"].append(0)
                        annotations[processed_sequence_name][frame_idx]["bbox"].append(person['bbox_ltwh'].tolist())
                        annotations[processed_sequence_name][frame_idx]["visibility"].append(1.0)
                        annotations[processed_sequence_name][frame_idx]["role"].append(role_mapping[person['role']])
                        annotations[processed_sequence_name][frame_idx]["legibility_score"].append(person['legibility_score'])
                        jn = person['jersey_number'] if person['legibility_score'] > 0.5 else None
                        
                        jn = str(int(jn)) if jn is not None else None
                        if (jn is not None) and (int(jn) < 0 or int(jn) > 99):
                            jn = None
                            
                        annotations[processed_sequence_name][frame_idx]["jersey"].append(jn_mapping[jn])
                        if jn is not None:
                            if len(jn) == 1:
                                annotations[processed_sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[jn])
                                annotations[processed_sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[None])
                            elif len(jn) == 2:
                                annotations[processed_sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[jn[0]])
                                annotations[processed_sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[jn[1]])
                            else:
                                annotations[processed_sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[None])
                                annotations[processed_sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[None])
                        else:
                            annotations[processed_sequence_name][frame_idx]["digit_head"].append(digit_head_mapping[None])
                            annotations[processed_sequence_name][frame_idx]["digit_tail"].append(digit_tail_mapping[None])
                            
                if soccer_factory_data[vid][frame_id]['valid_cam_params']:
                    K = soccer_factory_data[vid][frame_id]["K"]
                    R = soccer_factory_data[vid][frame_id]["R"]
                    P = soccer_factory_data[vid][frame_id]["P"]
                    annotations[processed_sequence_name][frame_idx]["K"] = K
                    annotations[processed_sequence_name][frame_idx]["R"] = R
                    annotations[processed_sequence_name][frame_idx]["P"] = P
                    annotations[processed_sequence_name][frame_idx]["valid_lines"] = True
                    annotations[processed_sequence_name][frame_idx]["valid_keypoints"] = True
                    annotations[processed_sequence_name][frame_idx]["lines"] = {}
                else:
                    annotations[processed_sequence_name][frame_idx]["valid_lines"] = False
                    annotations[processed_sequence_name][frame_idx]["valid_keypoints"] = False
                    annotations[processed_sequence_name][frame_idx]["lines"] = {}
                    
        for sequence_name in processed_sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                frame_annotation = annotations[sequence_name][i]
                if len(frame_annotation["id"]) > 0:
                    frame_annotation["id"] = torch.tensor(frame_annotation["id"], dtype=torch.int64)
                    frame_annotation["category"] = torch.tensor(frame_annotation["category"], dtype=torch.int64)
                    frame_annotation["bbox"] = torch.tensor(frame_annotation["bbox"], dtype=torch.float32)
                    frame_annotation["visibility"] = torch.tensor(frame_annotation["visibility"], dtype=torch.float32)
                    frame_annotation["role"] = torch.tensor(frame_annotation["role"], dtype=torch.int64)
                    frame_annotation["jersey"] = torch.tensor(frame_annotation["jersey"], dtype=torch.int64)
                    frame_annotation["digit_head"] = torch.tensor(frame_annotation["digit_head"], dtype=torch.int64)
                    frame_annotation["digit_tail"] = torch.tensor(frame_annotation["digit_tail"], dtype=torch.int64)
                    frame_annotation["legibility_score"] = torch.tensor(frame_annotation["legibility_score"], dtype=torch.float32)
                else:
                    # Empty frame
                    frame_annotation["id"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["category"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["bbox"] = torch.zeros((0, 4), dtype=torch.float32)
                    frame_annotation["visibility"] = torch.zeros((0, ), dtype=torch.float32)
                    frame_annotation["role"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["jersey"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["digit_head"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["digit_tail"] = torch.zeros((0, ), dtype=torch.int64)
                    frame_annotation["legibility_score"] = torch.zeros((0, ), dtype=torch.float32)

        for sequence_name in processed_sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name][i]["is_legal"] = is_legal(annotations[sequence_name][i])
        return annotations

    def _decouple_is_legal(self):
        decoupled_is_legal = defaultdict(list)
        for sequence_name in self.annotations:
            for frame_id, annotation in enumerate(self.annotations[sequence_name]):
                decoupled_is_legal[sequence_name].append(annotation["is_legal"])
        # Reformat the 'is_legal' attribute from a list to a tensor,
        # which is more convenient for the sampling process (calculation-friendly).
        decoupled_is_legal_in_tensor = defaultdict(torch.Tensor)
        for sequence_name in decoupled_is_legal:
            decoupled_is_legal_in_tensor[sequence_name] = torch.tensor(
                decoupled_is_legal[sequence_name], dtype=torch.bool
            )
        return decoupled_is_legal_in_tensor

    def set_sample_position(self):
        """
        Set the position of each legal sample.
        For test split in video mode, only frames where frame_idx % num_frames == 0 can be starting points.
        For train split in video mode with soccer_factory_data, only frames where frame_idx % num_frames == 0 can be starting points.
        Also ensures that starting position + num_frames doesn't exceed sequence length.
        """
        self.sample_position = list()
        for sequence_name in self.annotations:
            sequence_length = self.sequence_infos[sequence_name]["length"]
            is_soccer_factory_data = self.sequence_infos[sequence_name].get("is_soccer_factory_data", False)
            
            for frame_idx in range(len(self.annotations[sequence_name])):
                if self.annotations[sequence_name][frame_idx]["is_legal"]:
                    if (self.detection_data_type == "video" and 
                        self.backbone_type == "video" and 
                        self.split == "test"):
                        if (frame_idx % self.num_frames == 0 and 
                            frame_idx + self.num_frames <= sequence_length):
                            self.sample_position.append((sequence_name, frame_idx))
                    elif (self.detection_data_type == "video" and 
                        self.backbone_type == "video" and 
                        self.split == "train") and self.use_soccer_factory_data:
                        if (frame_idx % self.num_frames == 0 and 
                            frame_idx + self.num_frames <= sequence_length):
                            self.sample_position.append((sequence_name, frame_idx))
                    elif self.detection_data_type == "video" and self.backbone_type == "video":
                        if frame_idx + self.num_frames <= sequence_length:
                            self.sample_position.append((sequence_name, frame_idx))
                    else:
                        self.sample_position.append((sequence_name, frame_idx))
    
    def __len__(self):
        return len(self.sample_position)
    
    def format_data(self, image, annotation, metas):
        if self.transforms is not None:
            image, annotation, metas = self.transforms(image, annotation, metas)

        annotation['boxes'] = annotation['bbox']
        annotation['labels'] = annotation['category']
        annotation['roles'] = annotation['role']

        return image, annotation, metas
        
    
    def __getitem__(self, index):
        sequence_name, frame_idx = self.sample_position[index]
        
        if self.detection_data_type == "video" and self.backbone_type == "video":
            sequence_length = self.sequence_infos[sequence_name]["length"]
            start_frame = frame_idx
            end_frame = min(start_frame + self.num_frames, sequence_length)
            actual_num_frames = end_frame - start_frame
            
            images = []
            for i in range(start_frame, end_frame):
                image_path = self.image_paths[sequence_name][i]
                image = Image.open(image_path).convert("RGB")
                images.append(image)
            
            if sequence_name in self.soccer_factory_data_sequences:
                frame_indices = list(range(start_frame, end_frame))
                annotations = self._load_soccer_factory_data_frames(sequence_name, frame_indices)
            else:
                annotations = []
                for i in range(start_frame, end_frame):
                    annotation = copy.deepcopy(self.annotations[sequence_name][i])
                    annotations.append(annotation)
            
            metas = {"task": 'SoccerNetGSR_Detection',
                    "split": self.split,
                    "sequence": sequence_name,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "actual_num_frames": actual_num_frames,
                    "total_frames": self.num_frames,
                    "is_static": self.sequence_infos[sequence_name]["is_static"],
                    "size_divisibility": 1,}
            
            for i in range(len(images)):
                images[i], annotations[i], metas = self.format_data(images[i], annotations[i], metas)
                
            images = torch.stack(images, dim=0)
            
            return images, annotations, metas
        else:
            image_path = self.image_paths[sequence_name][frame_idx]
            image = Image.open(image_path).convert("RGB")
            
            if sequence_name in self.soccer_factory_data_sequences:
                annotations = self._load_soccer_factory_data_frames(sequence_name, [frame_idx])
                annotation = annotations[0]
            else:
                annotation = copy.deepcopy(self.annotations[sequence_name][frame_idx])
            
            metas = {"task": 'SoccerNetGSR_Detection',
                    "split": self.split,
                    "sequence": sequence_name,
                    "frame_idx": frame_idx,
                    "is_static": self.sequence_infos[sequence_name]["is_static"],
                    "size_divisibility": 1,}
            image, annotation, metas = self.format_data(image, annotation, metas)
            return image, annotation, metas

def build_gsr_detection_dataset(config: dict, split: str):
    assert 'SoccerNetGSR_Detection' in config['DATASETS_TO_HEADS'], "SoccerNetGSR_Detection must be in DATASETS_TO_HEADS"
    train_keypoints_or_lines_detection = 'LinesDetection' in config['DATASETS_TO_HEADS']['SoccerNetGSR_Detection'] or 'KeypointsDetection' in config['DATASETS_TO_HEADS']['SoccerNetGSR_Detection']
    
    dataset = SoccerNetGSR_Detection(
        data_root=config["DATA_ROOT"],
        sub_dir=config["SoccerNetGSR_SUB_DIR"],
        split=split,
        transforms=build_transforms(config, split),
        num_keypoints=config["NUM_KEYPOINTS"],
        num_lines=config["NUM_LINES"],
        detection_data_type=config["DETECTION_DATA_TYPE"],
        backbone_type=config["BACKBONE_TYPE"],
        num_frames=config["NUM_FRAMES"],
        image_input_size=config["AUG_MAX_SIZE"],
        detect_ball=config["DETR_DETECT_BALL"],
        detect_ball_only=config["DETECT_BALL_ONLY"],
        use_soccer_factory_data=config["USE_EXTRA_DATA"],
        soccer_factory_data_path=config["EXTRA_DATA_PATH"],
        soccer_factory_data_only=config["EXTRA_DATA_ONLY"],
        use_soccer_factory_data_amount=config["USE_EXTRA_DATA_AMOUNT"],
        train_keypoints_or_lines_detection=train_keypoints_or_lines_detection,
    )
    return dataset

def build_gsr_detection_dataloader(config: dict, split: str):
    dataset = build_gsr_detection_dataset(config, split)
    shuffle = True if split == "train" else False
    prefetch_factor = config["PREFETCH_FACTOR"] if config["NUM_WORKERS"] > 0 else None
    persistent_workers = config["NUM_WORKERS"] > 0
    return DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=shuffle, collate_fn=collate_fn, num_workers=config["NUM_WORKERS"], prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

def is_legal(annotation: dict):
    assert "id" in annotation, "Annotation must have 'id' field."
    assert "category" in annotation, "Annotation must have 'category' field."
    assert "bbox" in annotation, "Annotation must have 'bbox' field."
    assert "visibility" in annotation, "Annotation must have 'visibility' field."

    assert len(annotation["id"]) == len(annotation["category"]) \
           == len(annotation["bbox"]) == len(annotation["visibility"]), \
           "The length of 'id', 'category', 'bbox', 'visibility' must be the same."

    # assert torch.unique(annotation["id"]).size(0) == annotation["id"].size(0), f"IDs must be unique."
    _id_unique = torch.unique(annotation["id"]).size(0) == annotation["id"].size(0)     # for PersonPath22

    # A hack implementation for DETR (300 queries):
    # TODO: to make it more general, maybe pass the number of queries as an parameter.
    leq_300 = annotation["id"].shape[0] <= 300

    return len(annotation["id"]) > 0 and _id_unique and leq_300

def append_annotation(
        annotation: dict,
        obj_id: int,
        category: int,
        bbox: list,
        visibility: float,
):
    annotation["id"] = torch.cat([
        annotation["id"],
        torch.tensor([obj_id], dtype=torch.int64)
    ])
    annotation["category"] = torch.cat([
        annotation["category"],
        torch.tensor([category], dtype=torch.int64)
    ])
    annotation["bbox"] = torch.cat([
        annotation["bbox"],
        torch.tensor([bbox], dtype=torch.float32)
    ])
    annotation["visibility"] = torch.cat([
        annotation["visibility"],
        torch.tensor([visibility], dtype=torch.float32)
    ])
    return annotation

class BoxXYWHtoXYXY:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
        annotation["bbox"] = box_xywh_to_xyxy(annotation["bbox"])
        return image, annotation, metas


class BoxXYXYtoCXCYWH:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
        annotation["bbox"] = box_xyxy_to_cxcywh(annotation["bbox"])
        return image, annotation, metas

class BoxCXCYWHtoXYWH:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
        annotation["bbox"] = box_cxcywh_to_xywh(annotation["bbox"])
        return image, annotation, metas

class BoxXYWHtoCXCYWH:
    def __init__(self):
        return

    def __call__(self, image, annotation, metas):
        annotation["bbox"] = bbox_xywh_to_cxcywh(annotation["bbox"])
        return image, annotation, metas

class LRAmbiguityFix():
    def __init__(self, v_th=70, h_th=20):
        self.v_th = v_th
        self.h_th = h_th

    def __call__(self, image, annotation, metas):
        data = annotation['lines']

        if len(data) == 0:
            return image, annotation, metas

        n_left, n_right = self.compute_n_sides(data)

        angles_v, angles_h = [], []
        for line in data.keys():
            line_points = []
            for point in data[line]:
                line_points.append((point['x'], point['y']))

            sorted_points = sorted(line_points, key=lambda point: (point[0], point[1]))
            pi, pf = sorted_points[0], sorted_points[-1]
            if line in h_lines:
                angle_h = self.calculate_angle_h(pi[0], pi[1], pf[0], pf[1])
                if angle_h:
                    angles_h.append(abs(angle_h))
            if line in v_lines:
                angle_v = self.calculate_angle_v(pi[0], pi[1], pf[0], pf[1])
                if angle_v:
                    angles_v.append(abs(angle_v))


        if len(angles_h) > 0 and len(angles_v) > 0:
            if np.mean(angles_h) < self.h_th and np.mean(angles_v) < self.v_th:
                if n_right > n_left:
                    data = flip_annot_names(data, swap_top_bottom=False, swap_posts=False)
        annotation['lines'] = data

        return image, annotation, metas

    def calculate_angle_h(self, x1, y1, x2, y2):
        if not x2 - x1 == 0:
            slope = (y2 - y1) / (x2 - x1)
            angle = math.atan(slope)
            angle_degrees = math.degrees(angle)
            return angle_degrees
        else:
            return None
    def calculate_angle_v(self, x1, y1, x2, y2):
        if not x2 - x1 == 0:
            slope = (y2 - y1) / (x2 - x1)
            angle = math.atan(1 / slope) if slope != 0 else math.pi / 2  # Avoid division by zero
            angle_degrees = math.degrees(angle)
            return angle_degrees
        else:
            return None

    def compute_n_sides(self, data):
        n_left, n_right = 0, 0
        for line in data:
            line_words = line.split()[:3]
            if 'left' in line_words:
                n_left += 1
            elif 'right' in line_words:
                n_right += 1
        return n_left, n_right

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(v_th={self.v_th}, h_th={self.h_th})"


class KeypointsLinesDetectionTransform:
    def __init__(self, num_keypoints=58, num_lines=24, image_input_size=512):
        self.num_keypoints = num_keypoints
        self.num_lines = num_lines
        self.image_input_size = image_input_size

    def __call__(self, image, annotation, metas):
        
        try:
            if ('valid_lines' in annotation and annotation['valid_lines']) or 'valid_lines' not in annotation:
                line_db = LineKeypointsDB(annotation['lines'], image)
                lines_target = line_db.get_tensor()
                annotation['lines_target'] = torch.tensor(lines_target, dtype=torch.float32)
                annotation['valid_lines'] = torch.tensor(True, dtype=torch.bool)
            else:
                annotation['lines_target'] = torch.zeros((self.num_lines, self.image_input_size//2, self.image_input_size//2), dtype=torch.float32)
                annotation['valid_lines'] = torch.tensor(False, dtype=torch.bool)
        except Exception as e:
            annotation['lines_target'] = torch.zeros((self.num_lines, self.image_input_size//2, self.image_input_size//2), dtype=torch.float32)
            annotation['valid_lines'] = torch.tensor(False, dtype=torch.bool)
        
        
        if 'random_crop_params' in metas and metas['random_crop_apply']:
            params = metas['random_crop_params']
            crop_x, crop_y, crop_w, crop_h = params['crop_x'], params['crop_y'], params['crop_w'], params['crop_h']
            orig_w, orig_h = params['orig_w'], params['orig_h']
            max_dist_w = max(orig_w - crop_w - crop_x, crop_x)
            max_dist_h = max(orig_h - crop_h - crop_y, crop_y)
            extra_factor = max((max_dist_w + 0.5 * orig_w) / crop_w, (max_dist_h + 0.5 * orig_h) / crop_h)
        else:
            extra_factor = 0.5
        
        try:
            if ('valid_keypoints' in annotation and annotation['valid_keypoints']) or 'valid_keypoints' not in annotation:
                keypoints = KeypointsDB(annotation['lines'], image, extra_factor=extra_factor)
                keypoints_target, keypoints_mask = keypoints.get_tensor_w_mask()
                annotation['keypoints_target'] = torch.tensor(keypoints_target, dtype=torch.float32)
                annotation['keypoints_mask'] = torch.tensor(keypoints_mask, dtype=torch.float32)
                annotation['valid_keypoints'] = torch.tensor(True, dtype=torch.bool)
            else:
                annotation['keypoints_target'] = torch.zeros((self.num_keypoints, self.image_input_size//2, self.image_input_size//2), dtype=torch.float32)
                annotation['keypoints_mask'] = torch.zeros((self.num_keypoints), dtype=torch.float32)
                annotation['valid_keypoints'] = torch.tensor(False, dtype=torch.bool)
        except Exception as e:
            annotation['keypoints_target'] = torch.zeros((self.num_keypoints, self.image_input_size//2, self.image_input_size//2), dtype=torch.float32)
            annotation['keypoints_mask'] = torch.zeros((self.num_keypoints), dtype=torch.float32)
            annotation['valid_keypoints'] = torch.tensor(False, dtype=torch.bool)
        
        return image, annotation, metas

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_keypoints={self.num_keypoints}, num_lines={self.num_lines}, image_input_size={self.image_input_size})"

def build_transforms(config: dict, split: str = "train"):
    
    use_lr_ambiguity_fix = False
    use_keypoints_lines_detection = False
    if 'SoccerNetGSR_Detection' in config['DATASETS_TO_HEADS']:
        if 'LinesDetection' in config['DATASETS_TO_HEADS']['SoccerNetGSR_Detection'] or 'KeypointsDetection' in config['DATASETS_TO_HEADS']['SoccerNetGSR_Detection']:
            use_lr_ambiguity_fix = config['USE_LR_AMBIGUITY_FIX']
            use_keypoints_lines_detection = True
    
    transforms = [
        ClearAugmentationMetas(),
        LRAmbiguityFix() if use_lr_ambiguity_fix else None,
        ToTensor(),
    ]
    
    if split == "train" and config["AUG_ENABLE_TRAINING_AUGMENTATION"]:
        if config["AUG_ENABLE_RANDOM_AFFINE"]:
            transforms.append(RandomAffine(
                degrees=config["AUG_AFFINE_DEGREES"],
                translate=config["AUG_AFFINE_TRANSLATE"],
                scale=config["AUG_AFFINE_SCALE"],
                shear=config["AUG_AFFINE_SHEAR"],
                p=config["AUG_AFFINE_PROB"]
            ))
        
        if config["AUG_ENABLE_RANDOM_PERSPECTIVE"]:
            transforms.append(RandomPerspective(
                distortion_scale=config["AUG_PERSPECTIVE_DISTORTION_SCALE"],
                p=config["AUG_PERSPECTIVE_PROB"]
            ))
        
        if config["AUG_ENABLE_RANDOM_CROP"]:
            transforms.append(RandomCrop(
                crop_size_ratio_range=config["AUG_RANDOM_CROP_SIZE_RATIO_RANGE"],
                p=config["AUG_RANDOM_CROP_PROB"]
            ))
    
    transforms.append(RandomResize(sizes=config["AUG_RANDOM_RESIZE"], max_size=config["AUG_MAX_SIZE"], keep_aspect_ratio=config["KEEP_ASPECT_RATIO"]))
    
    if split == "train" and config["AUG_ENABLE_TRAINING_AUGMENTATION"]:
        if config["AUG_COLOR_JITTER_V2"]:
            transforms.append(ColorJitter(
                brightness=config["AUG_BRIGHTNESS"],
                contrast=config["AUG_CONTRAST"], 
                saturation=config["AUG_SATURATION"],
                hue=config["AUG_HUE"],
                p=1.0
            ))
        
        if config["AUG_RANDOM_HORIZONTAL_FLIP"]:
            transforms.append(RandomHorizontalFlip(p=config["AUG_HORIZONTAL_FLIP_PROB"]))
        
        if config["AUG_GAUSSIAN_NOISE"]:
            transforms.append(GaussianNoise(
                mean=0.0,
                std=config["AUG_GAUSSIAN_NOISE_STD"],
                p=config["AUG_GAUSSIAN_NOISE_PROB"]
            ))
        
        if config["AUG_GAUSSIAN_BLUR"]:
            transforms.append(GaussianBlur(
                kernel_size_range=config["AUG_GAUSSIAN_BLUR_KERNEL_SIZE_RANGE"],
                sigma_range=config["AUG_GAUSSIAN_BLUR_SIGMA_RANGE"],
                p=config["AUG_GAUSSIAN_BLUR_PROB"]
            ))
    
    transforms.extend([
        KeypointsLinesDetectionTransform(
            num_keypoints=config["NUM_KEYPOINTS"], 
            num_lines=config["NUM_LINES"], 
            image_input_size=config["AUG_MAX_SIZE"]
        ) if use_keypoints_lines_detection else None,
        Normalize(mean=config["AUG_MEAN"], std=config["AUG_STD"]),
        BoxXYWHtoCXCYWH(),
    ])
    
    return Compose(transforms)
    
def collate_fn(batch):
    images, annotations, metas = zip(*batch)
    _B = len(batch)
    images = torch.stack(images)
    
    return {
        "images": images,
        "annotations": annotations,
        "metas": metas,
    }
    
