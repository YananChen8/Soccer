import numpy as np
from collections import defaultdict
from ._base_metric import _BaseMetric
from .. import _timing

class YOLOmAP(_BaseMetric):
    """单类别AP计算（适用于person类别）"""

    def __init__(self, config=None):
        super().__init__()
        self.plottable = False
        self.iou_thresholds = np.linspace(0.5, 0.95, 10)  # COCO标准阈值
        self.iou_thresholds = np.append([0.01, 0.25], self.iou_thresholds) # 添加0.01和0.25阈值
        
        self.integer_array_list_fields = ['tp', 'fp']
        self.float_array_list_fields = ['confidence']
        
        self.integer_array_fields = ['TP', 'FN', 'FP']
        self.float_array_fields = ['APs', 'ARs']
        
        self.integer_fields = ['num_gt_dets', 'num_tracker_dets']
        self.float_fields = ['AP', 'AP0', 'AP25', 'AP50', 'AP75', 'AR', 'AR0', 'AR25', 'AR50', 'AR75']
        
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
        for field in self.integer_array_list_fields + self.float_array_list_fields:
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
            
            for t, (gt_dets_t, tracker_dets_t) in enumerate(zip(data['gt_dets'], data['tracker_dets'])):
                gt_dets_data_t = []
                tracker_dets_data_t = []
                
                for gt_id, gt_det in enumerate(gt_dets_t):
                    gt_dets_data_t.append({
                        'bbox_xywh': gt_det[-5:-1].astype(np.float32),
                        'used': False,
                        'gt_id': gt_id,
                        'match_tracker_id': -1
                    })
                for tracker_id, tracker_det in enumerate(tracker_dets_t):
                    tracker_dets_data_t.append({
                        'bbox_xywh': tracker_det[-5:-1].astype(np.float32),
                        'confidence': tracker_det[-1].astype(np.float32),
                        'tracker_id': tracker_id,
                        'match_gt_id': -1
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
                        tracker_det['tp'] = 1
                        tracker_det['fp'] = 0
                        tracker_det['match_gt_id'] = best_gt_id
                        gt_dets_data_t[best_gt_id]['used'] = True
                        gt_dets_data_t[best_gt_id]['match_tracker_id'] = tracker_det['tracker_id']
                    else:
                        tracker_det['tp'] = 0
                        tracker_det['fp'] = 1
                        # tracker_det['match_gt_id'] = -1 # 重复了
                        
                # 当前threshold下，聚合不同帧的匹配结果
                gt_dets_data.extend(gt_dets_data_t)
                tracker_dets_data.extend(tracker_dets_data_t)
                
            tp = np.array([det['tp'] for det in tracker_dets_data], dtype=float)
            fp = np.array([det['fp'] for det in tracker_dets_data], dtype=float)
            confidence = np.array([det['confidence'] for det in tracker_dets_data], dtype=float)
            
            res['tp'][thresh_idx] = tp
            res['fp'][thresh_idx] = fp
            res['confidence'][thresh_idx] = confidence
        
        

        return res
    
    def _compute_final_fields(self, res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        APs = []
        ARs = []
        TP = []
        FP = []
        FN = []
        for thresh_idx, iou_thresh in enumerate(self.iou_thresholds):
            tp = res['tp'][thresh_idx]
            fp = res['fp'][thresh_idx]
            confidence = res['confidence'][thresh_idx]
            
            # 按置信度排序
            sort_idx = np.argsort(-confidence)
            tp = tp[sort_idx]
            fp = fp[sort_idx]
            confidence = confidence[sort_idx]
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            total_gt = res['num_gt_dets']
            recalls = tp_cumsum / (total_gt + 1e-6)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            
            ap = self._compute_ap(precisions, recalls)
            APs.append(ap)
            
            max_recall = recalls[-1] if len(recalls) > 0 else 0.0
            ARs.append(max_recall)
            
            TP.append(tp_cumsum[-1])
            FP.append(fp_cumsum[-1])
            FN.append(total_gt - tp_cumsum[-1])
        res['APs'] = np.array(APs)
        res['ARs'] = np.array(ARs)
        
        res['TP'] = np.array(TP)
        res['FP'] = np.array(FP)
        res['FN'] = np.array(FN)

        res['AP0'] = res['APs'][0]
        res['AP25'] = res['APs'][1]
        res['AP50'] = res['APs'][2]
        res['AP75'] = res['APs'][7]
        res['AP'] = np.mean(res['APs'][2:]) # 从AP50开始计算
        
        res['AR0'] = res['ARs'][0]
        res['AR25'] = res['ARs'][1]
        res['AR50'] = res['ARs'][2]
        res['AR75'] = res['ARs'][7]
        res['AR'] = np.mean(res['ARs'][2:])  # 从AR50开始计算
        
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

    def _compute_ap(self, precisions, recalls):
        """使用COCO的101点插值法计算AP"""
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 101
        return ap
    
    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_array_list_fields + self.float_array_list_fields:
            res[field] = self._combine_concat(all_res, field)
        
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
            
        res = self._compute_final_fields(res)
        return res         

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = defaultdict(list)
        for class_name, class_res in all_res.items():
            for k, v in class_res.items():
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

# 使用示例 --------------------------------------------------
# 输入数据格式说明：
# gt_dets: List[np.array] 每个元素是Nx5数组（最后一维未使用）
# tracker_dets: List[np.array] 每个元素是Nx5数组（最后一维是confidence）

# # 初始化metric
# metric = mAPMetric()

# # 模拟数据
# data_example = {
#     'gt_dets': [
#         np.array([[10, 10, 20, 20, 0]]),  # 最后一位未使用
#         np.array([[50, 50, 30, 30, 0]]),
   

if __name__ == '__main__':
    # Create sample test data with pitch coordinates and image bbox
    # test_data = {
    #     'gt_dets': [
    #         [np.array([100, 100, 150, 100, 200, 100, 300, 400, 50, 60, 1.]),  # timestamp 1, det 1
    #          np.array([300, 300, 350, 300, 400, 300, 500, 600, 40, 45, 1.])],  # timestamp 1, det 2
    #         [np.array([150, 150, 200, 150, 250, 150, 350, 450, 55, 65, 1.])]   # timestamp 2, det 1
    #     ],
    #     'tracker_dets': [
    #         [np.array([110, 110, 160, 110, 210, 110, 305, 405, 50, 60, 0.99]),  # timestamp 1, det 1
    #          np.array([320, 320, 370, 320, 420, 320, 505, 605, 40, 45, 0.98])], # timestamp 1, det 2
    #         [np.array([140, 140, 190, 140, 240, 140, 355, 455, 55, 65, 0.97])]  # timestamp 2, det 1
    #     ]
    # }

    test_data = {
        'gt_dets': [
            [np.array([150, 150, 200, 150, 250, 150, 350, 450, 55, 65, 1.]),   # timestamp 1, gt 1
             np.array([150, 150, 200, 150, 250, 150, 250, 350, 55, 65, 1.])]   # timestamp 1, gt 1
        ],
        'tracker_dets': [
            [np.array([140, 140, 190, 140, 240, 140, 350, 450, 55, 65, 0.97]),  # timestamp 1, det 1
             np.array([140, 140, 190, 140, 240, 140, 450, 550, 55, 65, 0.96])]  # timestamp 1, det 1
        ]
    }

    # Initialize and run evaluation
    metric = YOLOmAP()
    results = metric.eval_sequence(test_data)
    print(results)
    # results = metric.combine_sequences({})
    # metric.print_table(results, "YOLO", "ALL")