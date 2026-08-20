import numpy as np
from collections import defaultdict
from ._base_metric import _BaseMetric
from .. import _timing

class AttributesACC(_BaseMetric):

    def __init__(self, config=None):
        super().__init__()
        self.plottable = False
        self.iou_thresholds = np.linspace(0.5, 0.95, 10)  # COCO标准阈值
        self.iou_thresholds = np.append([0.01, 0.25], self.iou_thresholds) # 添加0.01和0.25阈值
        
        self.integer_array_list_fields = []
        # self.float_array_list_fields = ['confidence', 'gt_role', 'tracker_role', 'gt_team', 'tracker_team', 'gt_jersey', 'tracker_jersey']
        self.float_array_list_fields = []
        self.bool_array_list_fields = ['match_role', 'match_team', 'match_jersey', 'role_mask', 'team_mask', 'jersey_mask']
        
        self.integer_array_fields = ['correct_role_cnt', 'total_role', 'correct_team_cnt', 'total_team', 'correct_jersey_cnt', 'total_jersey']
        self.float_array_fields = ['role_acc', 'team_acc', 'jersey_acc']
        
        self.integer_fields = []
        self.float_fields = ['RoleACC', 'RoleACC0', 'RoleACC25', 'RoleACC50', 'RoleACC75', 'TeamACC', 'TeamACC0', 'TeamACC25', 'TeamACC50', 'TeamACC75', 'JerseyACC', 'JerseyACC0', 'JerseyACC25', 'JerseyACC50', 'JerseyACC75']
        
        self.fields = self.float_array_fields + self.integer_fields + self.float_fields
        self.summary_fields = self.float_fields
        
        # self.gt_data = {}    # {frame_id: [{'bbox_xywh': [...], 'used': bool}, ...]}
        # self.det_data = {}   # {frame_id: [{'bbox_xywh': [...], 'confidence': float}, ...]}

    @_timing.time
    def eval_sequence(self, data):
        # data['gt_dets']: List[List[np.array]]
        # 长度表示有多少帧，每个元素是一个列表，包含多个numpy数组，每个数组表示一个GT框，格式如下：
        # [..., x, y, w, h, confidence]
        # 其中：
        # - (x, y) 是检测框的左上角顶点坐标
        # - w 是检测框的宽度
        # - h 是检测框的高度
        # - confidence 是检测框的置信度, gt值为1

        # data['tracker_dets']: List[List[np.array]]
        # 长度表示有多少帧，每个元素是一个列表，包含多个numpy数组，每个数组表示一个检测框，格式如下：
        # [..., x, y, w, h, confidence]
        # 其中：
        # - (x, y) 是检测框的左上角顶点坐标
        # - w 是检测框的宽度
        # - h 是检测框的高度
        # - confidence 是检测框的置信度
        
        res = {}
        for field in self.bool_array_list_fields + self.integer_array_list_fields + self.float_array_list_fields:
            res[field] = [None] * len(self.iou_thresholds)
            
        for field in self.integer_array_fields + self.float_array_fields:
            res[field] = np.zeros((len(self.iou_thresholds)), dtype=float)
            
        for field in self.integer_fields+ self.float_fields:
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
                        'team': gt_attributes_t['team'][gt_id],
                        'jersey': gt_attributes_t['jersey'][gt_id],
                        'role_mask': gt_attributes_t['role_mask'][gt_id],
                        'team_mask': gt_attributes_t['team_mask'][gt_id],
                        'jersey_mask': gt_attributes_t['jersey_mask'][gt_id],
                    })
                for tracker_id, tracker_det in enumerate(tracker_dets_t):
                    tracker_dets_data_t.append({
                        'bbox_xywh': tracker_det[-5:-1].astype(np.float32),
                        'confidence': tracker_det[-1].astype(np.float32),
                        'tracker_id': tracker_id,
                        'match_gt_id': -1,
                        'role': tracker_attributes_t['role'][tracker_id],
                        'team': tracker_attributes_t['team'][tracker_id],
                        'jersey': tracker_attributes_t['jersey'][tracker_id],
                        'role_mask': tracker_attributes_t['role_mask'][tracker_id],
                        'team_mask': tracker_attributes_t['team_mask'][tracker_id],
                        'jersey_mask': tracker_attributes_t['jersey_mask'][tracker_id],
                        'match_role': None,
                        'match_team': None,
                        'match_jersey': None,
                    })
                    
                # 当前threshold下，当前帧的匹配
                for tracker_det in tracker_dets_data_t:
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
                        # tracker_det['tp'] = 1
                        # tracker_det['fp'] = 0
                        tracker_det['match_gt_id'] = best_gt_id
                        gt_dets_data_t[best_gt_id]['used'] = True
                        gt_dets_data_t[best_gt_id]['match_tracker_id'] = tracker_det['tracker_id']
                        
                        # TODO: 要不要考虑gt_det的mask?
                        # if tracker_det['role_mask'] or gt_dets_data_t[best_gt_id]['role_mask']:
                        if tracker_det['role_mask']:
                            tracker_det['match_role'] = tracker_det['role'] == gt_dets_data_t[best_gt_id]['role']
                        
                        # if tracker_det['team_mask'] or gt_dets_data_t[best_gt_id]['team_mask']:
                        if tracker_det['team_mask']:
                            tracker_det['match_team'] = tracker_det['team'] == gt_dets_data_t[best_gt_id]['team']
                        
                        # if tracker_det['jersey_mask'] or gt_dets_data_t[best_gt_id]['jersey_mask']:
                        if tracker_det['jersey_mask']:
                            tracker_det['match_jersey'] = tracker_det['jersey'] == gt_dets_data_t[best_gt_id]['jersey']
                    else:
                        # match失败
                        tracker_det['role_mask'] = False
                        tracker_det['team_mask'] = False
                        tracker_det['jersey_mask'] = False
                        # tracker_det['tp'] = 0
                        # tracker_det['fp'] = 1
                        # tracker_det['match_gt_id'] = -1 # 重复了
                        
                # 当前threshold下，聚合不同帧的匹配结果
                gt_dets_data.extend(gt_dets_data_t)
                tracker_dets_data.extend(tracker_dets_data_t)
                
            # tp = np.array([det['tp'] for det in tracker_dets_data], dtype=float)
            # fp = np.array([det['fp'] for det in tracker_dets_data], dtype=float)
            # confidence = np.array([det['confidence'] for det in tracker_dets_data], dtype=float)
            match_role = np.array([det['match_role'] for det in tracker_dets_data])
            match_team = np.array([det['match_team'] for det in tracker_dets_data])
            match_jersey = np.array([det['match_jersey'] for det in tracker_dets_data])
            role_mask = np.array([det['role_mask'] for det in tracker_dets_data])
            team_mask = np.array([det['team_mask'] for det in tracker_dets_data])
            jersey_mask = np.array([det['jersey_mask'] for det in tracker_dets_data])
            
            # res['tp'][thresh_idx] = tp
            # res['fp'][thresh_idx] = fp
            # res['confidence'][thresh_idx] = confidence
            res['match_role'][thresh_idx] = match_role
            res['match_team'][thresh_idx] = match_team
            res['match_jersey'][thresh_idx] = match_jersey
            res['role_mask'][thresh_idx] = role_mask
            res['team_mask'][thresh_idx] = team_mask
            res['jersey_mask'][thresh_idx] = jersey_mask
        
        return res
    
    def _compute_final_fields(self, res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """

        correct_role_cnt = []
        total_role = []
        correct_team_cnt = []
        total_team = []
        correct_jersey_cnt = []
        total_jersey = []
        for thresh_idx, iou_thresh in enumerate(self.iou_thresholds):
            
            match_role = res['match_role'][thresh_idx]
            match_team = res['match_team'][thresh_idx]
            match_jersey = res['match_jersey'][thresh_idx]
            role_mask = res['role_mask'][thresh_idx]
            team_mask = res['team_mask'][thresh_idx]
            jersey_mask = res['jersey_mask'][thresh_idx]
            
            role_mask_true_idx = np.where(role_mask)[0]
            total_role.append(role_mask_true_idx.size)
            correct_role_cnt.append(np.sum(match_role[role_mask_true_idx]))
            
            team_mask_true_idx = np.where(team_mask)[0]
            total_team.append(team_mask_true_idx.size)
            correct_team_cnt.append(np.sum(match_team[team_mask_true_idx]))
            
            jersey_mask_true_idx = np.where(jersey_mask)[0]
            total_jersey.append(jersey_mask_true_idx.size)
            correct_jersey_cnt.append(np.sum(match_jersey[jersey_mask_true_idx]))
            
        res['correct_role_cnt'] = np.array(correct_role_cnt, dtype=float)
        res['total_role'] = np.array(total_role, dtype=float)
        res['correct_team_cnt'] = np.array(correct_team_cnt, dtype=float)
        res['total_team'] = np.array(total_team, dtype=float)
        res['correct_jersey_cnt'] = np.array(correct_jersey_cnt, dtype=float)
        res['total_jersey'] = np.array(total_jersey, dtype=float)
        
        res['role_acc'] = np.divide(res['correct_role_cnt'], res['total_role'], out=np.zeros_like(res['correct_role_cnt']), where=res['total_role']!=0)
        res['team_acc'] = np.divide(res['correct_team_cnt'], res['total_team'], out=np.zeros_like(res['correct_team_cnt']), where=res['total_team']!=0)
        res['jersey_acc'] = np.divide(res['correct_jersey_cnt'], res['total_jersey'], out=np.zeros_like(res['correct_jersey_cnt']), where=res['total_jersey']!=0)
        
        res['RoleACC0'] = res['role_acc'][0]
        res['RoleACC25'] = res['role_acc'][1]
        res['RoleACC50'] = res['role_acc'][2]
        res['RoleACC75'] = res['role_acc'][7]
        res['RoleACC'] = np.mean(res['role_acc'][2:])
        
        res['TeamACC0'] = res['team_acc'][0]
        res['TeamACC25'] = res['team_acc'][1]
        res['TeamACC50'] = res['team_acc'][2]
        res['TeamACC75'] = res['team_acc'][7]
        res['TeamACC'] = np.mean(res['team_acc'][2:])

        res['JerseyACC0'] = res['jersey_acc'][0]
        res['JerseyACC25'] = res['jersey_acc'][1]
        res['JerseyACC50'] = res['jersey_acc'][2]
        res['JerseyACC75'] = res['jersey_acc'][7]
        res['JerseyACC'] = np.mean(res['jersey_acc'][2:])
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
        # TODO: 能否压缩成一行代码？
        seqs = list(all_res.keys())
        num_thresholds = len(all_res[seqs[0]][field])
        res = []
        
        for threshold_idx in range(num_thresholds):
            full_list = []
            for seq in seqs:
                full_list.append(all_res[seq][field][threshold_idx]) # all_res[seq][field][threshold_idx]是一个numpy array
            full_list = np.concatenate(full_list, axis=0)
            res.append(full_list)
        
        return res