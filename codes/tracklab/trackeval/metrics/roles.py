import numpy as np
from collections import defaultdict
from ._base_metric import _BaseMetric
from .. import _timing

class RolesACC(_BaseMetric):

    def __init__(self, config=None):
        super().__init__()
        self.plottable = False
        self.iou_thresholds = np.linspace(0.5, 0.95, 10)  # COCO标准阈值
        self.iou_thresholds = np.append([0.01, 0.25], self.iou_thresholds) # 添加0.01和0.25阈值
        
        self.integer_array_list_fields = []
        self.float_array_list_fields = []
        self.bool_array_list_fields = ['match_role_P', 'match_role_RE', 'match_role_GK', 
                                     'role_mask_P', 'role_mask_RE', 'role_mask_GK']
        
        self.integer_array_fields = ['correct_role_P_cnt', 'total_role_P',
                                   'correct_role_RE_cnt', 'total_role_RE',
                                   'correct_role_GK_cnt', 'total_role_GK']
        self.float_array_fields = ['role_P_acc', 'role_RE_acc', 'role_GK_acc']
        
        self.integer_fields = []
        self.float_fields = ['RoleACC_P', 'RoleACC_P0', 'RoleACC_P25', 'RoleACC_P50', 'RoleACC_P75',
                           'RoleACC_RE', 'RoleACC_RE0', 'RoleACC_RE25', 'RoleACC_RE50', 'RoleACC_RE75',
                           'RoleACC_GK', 'RoleACC_GK0', 'RoleACC_GK25', 'RoleACC_GK50', 'RoleACC_GK75']
        
        self.fields = self.float_array_fields + self.integer_fields + self.float_fields
        self.summary_fields = self.float_fields

    @_timing.time
    def eval_sequence(self, data):
        res = {}
        for field in self.bool_array_list_fields + self.integer_array_list_fields + self.float_array_list_fields:
            res[field] = [None] * len(self.iou_thresholds)
            
        for field in self.integer_array_fields + self.float_array_fields:
            res[field] = np.zeros((len(self.iou_thresholds)), dtype=float)
            
        for field in self.integer_fields + self.float_fields:
            res[field] = 0
            
        res['num_gt_dets'] = data['num_gt_dets']
        res['num_tracker_dets'] = data['num_tracker_dets']
        
        for thresh_idx, iou_thresh in enumerate(self.iou_thresholds):
            gt_dets_data = []
            tracker_dets_data = []
            
            for t, (gt_dets_t, gt_attributes_t, tracker_dets_t, tracker_attributes_t) in enumerate(zip(data['gt_dets'], data['gt_attributes'], data['tracker_dets'], data['tracker_attributes'])):
                gt_dets_data_t = []
                tracker_dets_data_t = []
                
                for gt_id, gt_det in enumerate(gt_dets_t):
                    gt_dets_data_t.append({
                        'bbox_xywh': gt_det[-5:-1].astype(np.float32),
                        'used': False,
                        'gt_id': gt_id,
                        'match_tracker_id': -1,
                        'role': gt_attributes_t['role'][gt_id],
                        'role_mask': gt_attributes_t['role_mask'][gt_id],
                    })
                for tracker_id, tracker_det in enumerate(tracker_dets_t):
                    tracker_dets_data_t.append({
                        'bbox_xywh': tracker_det[-5:-1].astype(np.float32),
                        'confidence': tracker_det[-1].astype(np.float32),
                        'tracker_id': tracker_id,
                        'match_gt_id': -1,
                        'role': tracker_attributes_t['role'][tracker_id],
                        'role_mask': tracker_attributes_t['role_mask'][tracker_id],
                        'match_role_P': None,
                        'match_role_RE': None,
                        'match_role_GK': None,
                    })
                    
                # 当前threshold下，当前帧的匹配
                for tracker_det_idx, tracker_det in enumerate(tracker_dets_data_t):
                    best_iou = 0.0
                    best_gt_id = -1

                    for gt_det in gt_dets_data_t:
                        if gt_det['used']:
                            continue
                        iou = self._calculate_iou(tracker_det['bbox_xywh'], gt_det['bbox_xywh'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_id = gt_det['gt_id']
                            
                    if best_iou >= iou_thresh and best_gt_id != -1:
                        # match成功
                        tracker_det['match_gt_id'] = best_gt_id
                        gt_dets_data_t[best_gt_id]['used'] = True
                        gt_dets_data_t[best_gt_id]['match_tracker_id'] = tracker_det['tracker_id']
                        
                        if tracker_det['role_mask']:
                            gt_role = gt_dets_data_t[best_gt_id]['role']
                            pred_role = tracker_det['role']
                            
                            # 根据gt的role类型分别计算准确率
                            if gt_role == 'player':
                                tracker_det['match_role_P'] = pred_role == gt_role
                            elif gt_role == 'referee':
                                tracker_det['match_role_RE'] = pred_role == gt_role
                            elif gt_role == 'goalkeeper':
                                tracker_det['match_role_GK'] = pred_role == gt_role
                            else:
                                tracker_det['role_mask'] = False # TODO: 什么情况下会这样？
                    else:
                        # match失败
                        tracker_det['role_mask'] = False
                        
                # 当前threshold下，聚合不同帧的匹配结果
                gt_dets_data.extend(gt_dets_data_t)
                tracker_dets_data.extend(tracker_dets_data_t)
                
            match_role_P = np.array([det['match_role_P'] for det in tracker_dets_data])
            match_role_RE = np.array([det['match_role_RE'] for det in tracker_dets_data])
            match_role_GK = np.array([det['match_role_GK'] for det in tracker_dets_data])
            
            # role_mask = np.array([det['role_mask'] for det in tracker_dets_data])
            
            # # 根据gt的role类型设置对应的mask
            # gt_roles = []
            # for tracker_det in tracker_dets_data:
            #     if tracker_det['match_gt_id'] != -1:
            #         gt_roles.append(gt_dets_data[tracker_det['match_gt_id']]['role']) # 这当然有问题，match_gt_id只在那一帧生效
            #     else:
            #         gt_roles.append('none')
            # gt_roles = np.array(gt_roles)
            
            role_mask_P = (match_role_P == True) | (match_role_P == False)
            role_mask_RE = (match_role_RE == True) | (match_role_RE == False)
            role_mask_GK = (match_role_GK == True) | (match_role_GK == False)
            
            res['match_role_P'][thresh_idx] = match_role_P
            res['match_role_RE'][thresh_idx] = match_role_RE
            res['match_role_GK'][thresh_idx] = match_role_GK
            res['role_mask_P'][thresh_idx] = role_mask_P
            res['role_mask_RE'][thresh_idx] = role_mask_RE
            res['role_mask_GK'][thresh_idx] = role_mask_GK
            
            # print(len(tracker_dets_data), len(gt_dets_data), match_role_P.size, role_mask_P.size)
            # print(f"count: {np.sum(match_role_P[role_mask_P] == True) + np.sum(match_role_P[role_mask_P] == False)} {np.sum(role_mask_P == True)}")
            # exit()
        
        return res
    
    def _compute_final_fields(self, res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        correct_role_P_cnt = []
        total_role_P = []
        correct_role_RE_cnt = []
        total_role_RE = []
        correct_role_GK_cnt = []
        total_role_GK = []
        
        for thresh_idx, iou_thresh in enumerate(self.iou_thresholds):
            match_role_P = res['match_role_P'][thresh_idx]
            match_role_RE = res['match_role_RE'][thresh_idx]
            match_role_GK = res['match_role_GK'][thresh_idx]
            role_mask_P = res['role_mask_P'][thresh_idx]
            role_mask_RE = res['role_mask_RE'][thresh_idx]
            role_mask_GK = res['role_mask_GK'][thresh_idx]
            
            role_mask_P_true_idx = np.where(role_mask_P)[0]
            total_role_P.append(role_mask_P_true_idx.size)
            correct_role_P_cnt.append(np.sum(match_role_P[role_mask_P_true_idx]))
            
            role_mask_RE_true_idx = np.where(role_mask_RE)[0]
            total_role_RE.append(role_mask_RE_true_idx.size)
            correct_role_RE_cnt.append(np.sum(match_role_RE[role_mask_RE_true_idx]))
            
            role_mask_GK_true_idx = np.where(role_mask_GK)[0]
            total_role_GK.append(role_mask_GK_true_idx.size)
            correct_role_GK_cnt.append(np.sum(match_role_GK[role_mask_GK_true_idx]))
            
        res['correct_role_P_cnt'] = np.array(correct_role_P_cnt, dtype=float)
        res['total_role_P'] = np.array(total_role_P, dtype=float)
        res['correct_role_RE_cnt'] = np.array(correct_role_RE_cnt, dtype=float)
        res['total_role_RE'] = np.array(total_role_RE, dtype=float)
        res['correct_role_GK_cnt'] = np.array(correct_role_GK_cnt, dtype=float)
        res['total_role_GK'] = np.array(total_role_GK, dtype=float)
        
        res['role_P_acc'] = np.divide(res['correct_role_P_cnt'], res['total_role_P'], out=np.zeros_like(res['correct_role_P_cnt']), where=res['total_role_P']!=0)
        res['role_RE_acc'] = np.divide(res['correct_role_RE_cnt'], res['total_role_RE'], out=np.zeros_like(res['correct_role_RE_cnt']), where=res['total_role_RE']!=0)
        res['role_GK_acc'] = np.divide(res['correct_role_GK_cnt'], res['total_role_GK'], out=np.zeros_like(res['correct_role_GK_cnt']), where=res['total_role_GK']!=0)
        
        res['RoleACC_P0'] = res['role_P_acc'][0]
        res['RoleACC_P25'] = res['role_P_acc'][1]
        res['RoleACC_P50'] = res['role_P_acc'][2]
        res['RoleACC_P75'] = res['role_P_acc'][7]
        res['RoleACC_P'] = np.mean(res['role_P_acc'][2:])
        
        res['RoleACC_RE0'] = res['role_RE_acc'][0]
        res['RoleACC_RE25'] = res['role_RE_acc'][1]
        res['RoleACC_RE50'] = res['role_RE_acc'][2]
        res['RoleACC_RE75'] = res['role_RE_acc'][7]
        res['RoleACC_RE'] = np.mean(res['role_RE_acc'][2:])
        
        res['RoleACC_GK0'] = res['role_GK_acc'][0]
        res['RoleACC_GK25'] = res['role_GK_acc'][1]
        res['RoleACC_GK50'] = res['role_GK_acc'][2]
        res['RoleACC_GK75'] = res['role_GK_acc'][7]
        res['RoleACC_GK'] = np.mean(res['role_GK_acc'][2:])
        
        return res

    def _xywh_to_xyxy(self, bbox):
        """将xywh格式转换为xyxy格式"""
        x, y, w, h = bbox
        return [x, y, x + w, y + h]

    def _calculate_iou(self, box1, box2):
        """计算两个xyxy格式框的IoU"""
        # 转换坐标
        box1 = self._xywh_to_xyxy(box1)
        box2 = self._xywh_to_xyxy(box2)
        
        # 计算交集
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算并集
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area >0 else 0
    
    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.bool_array_list_fields:
            res[field] = self._combine_concat(all_res, field)
        
        res = self._compute_final_fields(res)
        return res         

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = defaultdict(list)
        for class_name, class_res in all_res.items():
            for k, v in class_res.items():
                if k in self.float_fields:
                    res[k].append(v)
        for k, v in res.items():
            res[k] = np.array(v)
            res[k] = np.mean(res[k])
        
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = defaultdict(list)
        for class_name, class_res in all_res.items():
            for k, v in class_res.items():
                if k in self.float_fields:
                    res[k].append(v)
        for k, v in res.items():
            res[k] = np.array(v)
            res[k] = np.mean(res[k])
        
        return res
    
    @staticmethod
    def _combine_concat(all_res, field):
        seqs = list(all_res.keys())
        num_thresholds = len(all_res[seqs[0]][field])
        res = []
        
        for threshold_idx in range(num_thresholds):
            full_list = []
            for seq in seqs:
                full_list.append(all_res[seq][field][threshold_idx])
            full_list = np.concatenate(full_list, axis=0)
            res.append(full_list)
        
        return res