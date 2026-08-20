from functools import partial
from pathlib import Path
from typing import Any
from PIL import Image

import os
import sys
import yaml
import copy
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.download import download_file
from tracklab.pipeline.videolevel_module import VideoLevelModule

from nbjw_calib.model.cls_hrnet import get_cls_net
from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
from nbjw_calib.utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool, \
                                            get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, \
                                            coords_to_dict)
from nbjw_calib.utils.utils_calib import FramebyFrameCalib
from scipy.signal import savgol_filter

def kp_to_line(keypoints):
    line_keypoints_match = {"Big rect. left bottom": [24, 68, 25],
                            "Big rect. left main": [5, 64, 31, 46, 34, 66, 25],
                            "Big rect. left top": [4, 62, 5],
                            "Big rect. right bottom": [26, 69, 27],
                            "Big rect. right main": [6, 65, 33, 56, 36, 67, 26],
                            "Big rect. right top": [6, 63, 7],
                            "Circle central": [32, 48, 38, 50, 42, 53, 35, 54, 43, 52, 39, 49],
                            "Circle left": [31,37, 47, 41, 34],
                            "Circle right": [33, 40, 55, 44, 36],
                            "Goal left crossbar": [16, 12],
                            "Goal left post left": [16, 17],
                            "Goal left post right": [12, 13],
                            "Goal right crossbar": [15, 19],
                            "Goal right post left": [15, 14],
                            "Goal right post right": [19, 18],
                            "Middle line": [2, 32, 51, 35, 29],
                            "Side line bottom": [28, 70, 71, 29, 72, 73, 30],
                            "Side line left": [1, 4, 8, 13,17, 20, 24, 28],
                            "Side line right": [3, 7, 11, 14, 18, 23, 27, 30],
                            "Side line top": [1, 58, 59, 2, 60, 61, 3],
                            "Small rect. left bottom": [20, 21],
                            "Small rect. left main": [9, 21],
                            "Small rect. left top": [8, 9],
                            "Small rect. right bottom": [22, 23],
                            "Small rect. right main": [10, 22],
                            "Small rect. right top": [10, 11]}

    lines = {}
    for line_name, kp_indices in line_keypoints_match.items():
        line = []
        for idx in kp_indices:
            if idx in keypoints.keys():
                line.append({'x': keypoints[idx]['x'], 'y': keypoints[idx]['y']})

        if line:
            lines[line_name] = line

    return lines

class NBJW_Calib_Keypoints(ImageLevelModule):

    input_columns = {
        "image": [],
        "detection": [],
    }
    output_columns = {
        "image": ["keypoints", "lines"],
        "detection": []
    }

    def __init__(self, checkpoint_kp, checkpoint_l, image_width, image_height, batch_size, device, cfg, cfg_l, **kwargs):
        super().__init__(batch_size)
        self.device = device

        self.cfg = cfg
        self.cfg_l = cfg_l

        if not os.path.isfile(checkpoint_kp):
            download_file("https://zenodo.org/records/12626395/files/SV_kp?download=1", checkpoint_kp)

        if not os.path.isfile(checkpoint_l):
            download_file("https://zenodo.org/records/12626395/files/SV_lines?download=1", checkpoint_l)


        loaded_state = torch.load(checkpoint_kp, map_location=device)
        self.model = get_cls_net(self.cfg)
        self.model.load_state_dict(loaded_state)
        self.model.to(device)
        self.model.eval()

        loaded_state_l = torch.load(checkpoint_l, map_location=device)
        self.model_l = get_cls_net_l(self.cfg_l)
        self.model_l.load_state_dict(loaded_state_l)
        self.model_l.to(device)
        self.model_l.eval()

        self.tfms_resize = T.Compose(
            [T.Resize((540, 960)),
             T.ToTensor()])

        self.tfms = T.ToTensor()

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        image = Image.fromarray(image).convert("RGB")
        image = self.tfms_resize(image)
        #image = self.tfms(image)
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):

        with torch.no_grad():
            heatmaps = self.model(batch.to(self.device))
            heatmaps_l = self.model_l(batch.to(self.device))

        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        kp_dict = coords_to_dict(kp_coords, threshold=0.1449)
        lines_dict = coords_to_dict(line_coords, threshold=0.2983)

        image_width = batch.size()[-1]
        image_height = batch.size()[-2]
        final_dict = complete_keypoints(kp_dict, lines_dict, w=image_width, h=image_height, normalize=True)

        output_pred = []
        for result, idx in zip(final_dict, metadatas.index):
            output_pred.append(pd.Series({"keypoints": result, "lines": kp_to_line(result)}, name=idx,))


        return pd.DataFrame(),  pd.DataFrame(output_pred)


    def flatten_dict(self, d):
        flat_dict = {}
        for outer_key, inner_dict in d.items():
            for inner_key, value in inner_dict.items():
                flat_dict[f"{outer_key}_{inner_key}"] = value
        return flat_dict

    def reconstruct_dict(self, row, original_keys):
        new_dict = {}
        for key in original_keys:
            sub_dict = {k.split('_')[1]: row[k] for k in row.index if k.startswith(f"{key}_") and pd.notna(row[k])}
            if sub_dict:  # Only add sub_dict if it is not empty
                new_dict[int(key)] = sub_dict
        return new_dict


class NBJW_Calib(ImageLevelModule):
    input_columns = {
        "image": ["keypoints"],
        "detection": ["bbox_ltwh"],
    }
    output_columns = {
        "image": ["parameters"],
        "detection": ["bbox_pitch"],
    }

    def __init__(self, image_width, image_height, batch_size, use_prev_homography, **kwargs):
        super().__init__(batch_size)
        self.image_width = image_width
        self.image_height = image_height
        self.cam = FramebyFrameCalib(self.image_width, self.image_height, denormalize=True)
        self.use_prev_homography = use_prev_homography

        self.last_h = None
        self.last_params = None

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        predictions = metadatas["keypoints"][0]

        self.cam.update(predictions)
        h = self.cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)
        if self.use_prev_homography:
            if h is not None:
                camera_predictions = self.cam.heuristic_voting()["cam_params"]
                detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch_h(h))
                self.last_h = h
                self.last_params = camera_predictions
            else:
                if self.last_h is not None:
                    camera_predictions = self.last_params
                    h = self.last_h
                    detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch_h(h))
                else:
                    camera_predictions = {}
                    detections["bbox_pitch"] = None
            return detections[["bbox_pitch"]], pd.DataFrame([
                pd.Series({"parameters": camera_predictions}, name=metadatas.iloc[0].name)
            ])
        else:
            if h is not None:
                camera_predictions = self.cam.heuristic_voting()['cam_params']
                detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch_h(h))
            else:
                camera_predictions = {}
                detections["bbox_pitch"] = None

            return detections[["bbox_pitch"]], pd.DataFrame([
                pd.Series({"parameters": camera_predictions}, name=metadatas.iloc[0].name)
            ])
            
class NBJW_Calib_Decoupled(ImageLevelModule):
    input_columns = {
        "image": ["keypoints"],
        "detection": [],
    }
    output_columns = {
        "image": ["parameters", "h"],
        "detection": [],
    }

    def __init__(self, image_width, image_height, batch_size=None, **kwargs):
        super().__init__(batch_size=1)
        self.image_width = image_width
        self.image_height = image_height
        self.cam = FramebyFrameCalib(self.image_width, self.image_height, denormalize=True)

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        predictions = metadatas["keypoints"][0]

        self.cam.update(predictions)
        h = self.cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)

        if h is not None:
            voting_result = self.cam.heuristic_voting()
            if voting_result is not None:
                camera_predictions = voting_result['cam_params']
            else:
                camera_predictions = {}
        else:
            camera_predictions = {}

        return pd.DataFrame(),  pd.DataFrame([
            pd.Series({"parameters": camera_predictions, "h": h}, name=metadatas.iloc[0].name)
        ])

class ApplyParameters(ImageLevelModule):
    input_columns = {
        "image": ["parameters", "h"],
        "detection": ["bbox_ltwh"],
    }
    output_columns = {
        "detection": ["bbox_pitch"],
    }

    def __init__(self, use_h, use_linalg, use_prev_homography, batch_size=None, **kwargs):
        super().__init__(batch_size=1)
        self.use_h = use_h
        self.use_linalg = use_linalg
        self.use_prev_homography = use_prev_homography

        self.last_h = None
        self.last_params = {}
        
        self.reset()

    def reset(self):
        self.last_h = None
        self.last_params = {}

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        h = metadatas["h"][0]
        cam_params = metadatas["parameters"][0]
        
        # Use previous homography/params if current ones are not available
        if self.use_prev_homography and ((self.use_h and (h is None or np.any(np.isnan(h)))) or (not self.use_h and 'x_focal_length' not in cam_params)):
            h = self.last_h
            cam_params = self.last_params
            
        # Process if we have valid parameters
        if (self.use_h and h is not None and not np.any(np.isnan(h))) or (not self.use_h and 'x_focal_length' in cam_params):
            # Calculate projection matrix or K,R,T if not using homography
            if not self.use_h:
                if self.use_linalg:
                    P = projection_from_cam_params(cam_params)
                else:
                    K, R, T = get_KRT_from_cam_params(cam_params)
            
            # Calculate bbox_pitch based on parameters
            if self.use_h:
                detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch_h(h))
            else:
                if self.use_linalg:
                    detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch_cam_params_linalg(P))
                else:
                    detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch_cam_params(K, R, T))
                    
            # Store current parameters if using previous homography
            if self.use_prev_homography:
                self.last_h = h
                self.last_params = cam_params
        else:
            # No valid parameters available
            detections["bbox_pitch"] = None

        return detections[["bbox_pitch"]]
        
class GetPitchCorners(ImageLevelModule):
    input_columns = {
        "image": ["parameters", "h"],
    }
    output_columns = {
        "image": ["pitch_corners"],
    }

    def __init__(self, use_h, use_linalg, use_prev_homography, image_width, image_height, batch_size=None, **kwargs):
        super().__init__(batch_size=1)
        self.use_h = use_h
        self.use_linalg = use_linalg
        self.use_prev_homography = use_prev_homography
        self.image_width = image_width
        self.image_height = image_height

        self.last_h = None
        self.last_params = None
        
        self.reset()

    def reset(self):
        self.last_h = None
        self.last_params = None

    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        return image

    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        h = metadatas["h"][0]
        cam_params = metadatas["parameters"][0]
        image_corners = np.array([[0, 0], [self.image_width, 0], [self.image_width, self.image_height], [0, self.image_height]]) # 左上，右上，右下，左下
        
        # Use previous homography/params if current ones are not available
        if self.use_prev_homography and ((self.use_h and h is None) or (not self.use_h and 'x_focal_length' not in cam_params)):
            h = self.last_h
            cam_params = self.last_params
            
        # Process if we have valid parameters
        if (self.use_h and h is not None) or (not self.use_h and 'x_focal_length' in cam_params):
            # Calculate projection matrix or K,R,T if not using homography
            if not self.use_h:
                if self.use_linalg:
                    P = projection_from_cam_params(cam_params)
                else:
                    K, R, T = get_KRT_from_cam_params(cam_params)
            
            if self.use_h:
                pitch_corners = np.stack([get_bbox_pitch_point_h(h, point) for point in image_corners])
            else:
                if self.use_linalg:
                    pitch_corners = np.stack([get_bbox_pitch_point_cam_params_linalg(P, point) for point in image_corners])
                else:
                    pitch_corners = np.stack([get_bbox_pitch_point_cam_params(K, R, T, point) for point in image_corners])
                    
            # Store current parameters if using previous homography
            if self.use_prev_homography:
                self.last_h = h
                self.last_params = cam_params
        else:
            # No valid parameters available
            pitch_corners = None

        return pd.DataFrame(), pd.DataFrame([
            pd.Series({"pitch_corners": pitch_corners}, name=metadatas.iloc[0].name)
        ])


def get_bbox_pitch_point_h(h, point):
    unproj_point = h @ np.array([point[0], point[1], 1])
    unproj_point /= unproj_point[2]

    pitch_x, pitch_y, _ = unproj_point
    if np.any(np.isnan([pitch_x, pitch_y])):
        return None
    return np.array([pitch_x, pitch_y])

def get_bbox_pitch_point_cam_params_linalg(P, point):
    u, v = point[0], point[1]
    # 构建系数矩阵
    A = np.array([
        [u*P[2,0] - P[0,0], u*P[2,1] - P[0,1]],  # 第一行方程系数
        [v*P[2,0] - P[1,0], v*P[2,1] - P[1,1]]   # 第二行方程系数
    ])
    
    # 构建常数项
    B = np.array([
        P[0,3] - u*P[2,3],  # 第一方程常数项
        P[1,3] - v*P[2,3]   # 第二方程常数项
    ])

    try:
        XY = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return None  # 处理奇异矩阵情况
        
    return np.array([XY[0], XY[1]])

def get_bbox_pitch_point_cam_params(K, R, T, point):
    u, v = point
    
    Pi_homogeneous = np.array([u, v, 1.0], dtype=np.float64)
    
    K_inv = np.linalg.inv(K)                # 内参逆矩阵
    P_cam_normalized = K_inv @ Pi_homogeneous  # 相机坐标系归一化坐标
    R_inv = np.linalg.inv(R)                # 旋转逆矩阵
    ray_dir_world = R_inv @ P_cam_normalized # 世界坐标系射线方向
    
    # 处理光线与平面平行的情况
    # if np.isclose(ray_dir_world[2], 0):
    #     return None
    
    s = (-T[2]) / ray_dir_world[2]
    
    X = T[0] + s * ray_dir_world[0]
    Y = T[1] + s * ray_dir_world[1]
    
    return np.array([X, Y])

def get_bbox_pitch_h(h):
    def unproject_point_on_planeZ0(h, point):
        unproj_point = h @ np.array([point[0], point[1], 1])
        unproj_point /= unproj_point[2]
        return unproj_point

    def _get_bbox(bbox_ltrb):
        l, t, r, b = bbox_ltrb
        bl = [l, b]
        br = [r, b]
        bm = [l+(r-l)/2, b]
        pbl_x, pbl_y, _ = unproject_point_on_planeZ0(h, bl)
        pbr_x, pbr_y, _ = unproject_point_on_planeZ0(h, br)
        pbm_x, pbm_y, _ = unproject_point_on_planeZ0(h, bm)
        if np.any(np.isnan([pbl_x, pbl_y, pbr_x, pbr_y, pbm_x, pbm_y])):
            return None
        return {
            "x_bottom_left": pbl_x, "y_bottom_left": pbl_y,
            "x_bottom_right": pbr_x, "y_bottom_right": pbr_y,
            "x_bottom_middle": pbm_x, "y_bottom_middle": pbm_y,
        }
    return _get_bbox

def get_bbox_pitch_cam_params_linalg(P):
    def unproject_point_on_planeZ0(P, point):
        u, v = point[0], point[1]
        # 构建系数矩阵
        A = np.array([
            [u*P[2,0] - P[0,0], u*P[2,1] - P[0,1]],  # 第一行方程系数
            [v*P[2,0] - P[1,0], v*P[2,1] - P[1,1]]   # 第二行方程系数
        ])
        
        # 构建常数项
        B = np.array([
            P[0,3] - u*P[2,3],  # 第一方程常数项
            P[1,3] - v*P[2,3]   # 第二方程常数项
        ])

        try:
            XY = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            return None  # 处理奇异矩阵情况
        
        return np.array([XY[0], XY[1], 0.0])

    def _get_bbox(bbox_ltrb):
        l, t, r, b = bbox_ltrb
        bl = [l, b]
        br = [r, b]
        bm = [l+(r-l)/2, b]
        pbl_x, pbl_y, _ = unproject_point_on_planeZ0(P, bl)
        pbr_x, pbr_y, _ = unproject_point_on_planeZ0(P, br)
        pbm_x, pbm_y, _ = unproject_point_on_planeZ0(P, bm)
        if np.any(np.isnan([pbl_x, pbl_y, pbr_x, pbr_y, pbm_x, pbm_y])):
            return None
        return {
            "x_bottom_left": pbl_x, "y_bottom_left": pbl_y,
            "x_bottom_right": pbr_x, "y_bottom_right": pbr_y,
            "x_bottom_middle": pbm_x, "y_bottom_middle": pbm_y,
        }
    return _get_bbox

def get_bbox_pitch_cam_params(K, R, T):
    def backproject_using_ray_method(K, R, T, image_point):
        """
        通过射线相交法将图像点反投影到世界坐标系z=0平面
        
        Args:
            image_point (tuple): 图像坐标 (u, v)
            K (np.ndarray): 3x3内参矩阵
            R (np.ndarray): 3x3旋转矩阵
            T (np.ndarray): 3D平移向量（相机位置）
        
        Returns:
            np.ndarray: 世界坐标系点 [X, Y, 0.0]，无交点返回None
        """
        u, v = image_point
        
        # 转换为齐次坐标 (3D向量)
        Pi_homogeneous = np.array([u, v, 1.0], dtype=np.float64)
        
        # 计算射线方向（分步验证）
        K_inv = np.linalg.inv(K)                # 内参逆矩阵
        P_cam_normalized = K_inv @ Pi_homogeneous  # 相机坐标系归一化坐标
        R_inv = np.linalg.inv(R)                # 旋转逆矩阵
        ray_dir_world = R_inv @ P_cam_normalized # 世界坐标系射线方向
        
        # 处理光线与平面平行的情况
        # if np.isclose(ray_dir_world[2], 0):
        #     return None
        
        # 计算射线参数 (s = (0 - T_z) / dir_z)
        s = (-T[2]) / ray_dir_world[2]
        
        # 计算交点坐标
        X = T[0] + s * ray_dir_world[0]
        Y = T[1] + s * ray_dir_world[1]
        
        return np.array([X, Y, 0.0])

    def _get_bbox(bbox_ltrb):
        l, t, r, b = bbox_ltrb
        bl = [l, b]
        br = [r, b]
        bm = [l+(r-l)/2, b]
        pbl_x, pbl_y, _ = backproject_using_ray_method(K, R, T, bl)
        pbr_x, pbr_y, _ = backproject_using_ray_method(K, R, T, br)
        pbm_x, pbm_y, _ = backproject_using_ray_method(K, R, T, bm)
        if np.any(np.isnan([pbl_x, pbl_y, pbr_x, pbr_y, pbm_x, pbm_y])):
            return None
        return {
            "x_bottom_left": pbl_x, "y_bottom_left": pbl_y,
            "x_bottom_right": pbr_x, "y_bottom_right": pbr_y,
            "x_bottom_middle": pbm_x, "y_bottom_middle": pbm_y,
        }
    return _get_bbox

def get_KRT_from_cam_params(cam_params):
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'])
    position_meters = np.array(cam_params['position_meters'])
    rotation = np.array(cam_params['rotation_matrix'])
    
    T = position_meters
    R = rotation
    K = np.array([[x_focal_length, 0, principal_point[0]],
                  [0, y_focal_length, principal_point[1]],
                  [0, 0, 1]])
    return K, R, T

def projection_from_cam_params(cam_params):
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'])
    position_meters = np.array(cam_params['position_meters'])
    rotation = np.array(cam_params['rotation_matrix'])

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([[x_focal_length, 0, principal_point[0]],
                  [0, y_focal_length, principal_point[1]],
                  [0, 0, 1]])
    P = Q @ (rotation @ It)

    return P