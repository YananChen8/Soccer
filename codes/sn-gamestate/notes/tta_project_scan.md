# TTA Project Scan

- date_remote: 2026-06-27T23:28:50-04:00
- root: /remote-home/jiayuanrao/yishan/SoccerMaster/codes/sn-gamestate
- python: /usr/bin/python3
- output_root: outputs/tta_calib

## Plan

1. Scan NBJW/PnLCalib inputs, outputs, heatmaps, solvers, and evaluators.
2. Recompute baseline diagnostics from cached heatmaps without TTA.
3. Only after baseline is reproducible, add camera-level TTA and gate.

## Candidate Entry Files

- experiments/detection_benchmark/analyze_ransac_outlier_tracks_full49.py
- experiments/detection_benchmark/broadtrack_min_ablation_round3.py
- experiments/detection_benchmark/build_temporal_dataset.py
- experiments/detection_benchmark/cached_full_test_round2.py
- experiments/detection_benchmark/cached_outlier_only_eval_round3.py
- experiments/detection_benchmark/cache_hrnet_pred_only.py
- experiments/detection_benchmark/dense_flow_init_ablation_round3.py
- experiments/detection_benchmark/eval_temporal_feature_fusion_calib.py
- experiments/detection_benchmark/eval_temporal_input_mixer_calib.py
- experiments/detection_benchmark/eval_train12_cached_token_metrics.py
- experiments/detection_benchmark/export_ransac_outlier_k5_point_cases.py
- experiments/detection_benchmark/fast_full_test.py
- experiments/detection_benchmark/inject_outlier_params_to_sam3_state_round3.py
- experiments/detection_benchmark/merge_full_eval.py
- experiments/detection_benchmark/__pycache__/broadtrack_min_ablation_round3.cpython-310.pyc
- experiments/detection_benchmark/__pycache__/cached_full_test_round2.cpython-310.pyc
- experiments/detection_benchmark/__pycache__/eval_temporal_feature_fusion_calib.cpython-310.pyc
- experiments/detection_benchmark/__pycache__/eval_temporal_input_mixer_calib.cpython-310.pyc
- experiments/detection_benchmark/__pycache__/inject_outlier_params_to_sam3_state_round3.cpython-310.pyc
- experiments/detection_benchmark/__pycache__/standalone_calib_eval.cpython-310.pyc
- experiments/detection_benchmark/__pycache__/train_temporal_adapter_online.cpython-310.pyc
- experiments/detection_benchmark/__pycache__/train_temporal_feature_fusion_smoke.cpython-310.pyc
- experiments/detection_benchmark/__pycache__/train_temporal_input_mixer_smoke.cpython-310.pyc
- experiments/detection_benchmark/__pycache__/visualize_broadtrack_ablation_round3.cpython-310.pyc
- experiments/detection_benchmark/render_broadtrack_bev_video_round3.py
- experiments/detection_benchmark/render_broadtrack_bev_with_image_round3.py
- experiments/detection_benchmark/smoke_inference_temporal.py
- experiments/detection_benchmark/standalone_calib_eval_full.py
- experiments/detection_benchmark/standalone_calib_eval.py
- experiments/detection_benchmark/train_temporal_adapter_online.py
- experiments/detection_benchmark/train_temporal_adapter.py
- experiments/detection_benchmark/train_temporal_feature_fusion_smoke.py
- experiments/detection_benchmark/train_temporal_input_mixer_smoke.py
- experiments/detection_benchmark/train_temporal_token_adapter_cached.py
- experiments/detection_benchmark/train_temporal_token_adapter_online.py
- experiments/detection_benchmark/visualize_broadtrack_ablation_round3.py
- experiments/detection_benchmark/visualize_broadtrack_overlays_from_csv_round3.py
- experiments/detection_benchmark/visualize_flow_recovery_case_round3.py
- plugins/calibration/pyproject.toml
- plugins/calibration/sn_calibration_baseline/baseline_cameras.py
- plugins/calibration/sn_calibration_baseline/camera.py
- plugins/calibration/sn_calibration_baseline/dataloader.py
- plugins/calibration/sn_calibration_baseline/detect_extremities.py
- plugins/calibration/sn_calibration_baseline/evalai_camera.py
- plugins/calibration/sn_calibration_baseline/evaluate_camera.py
- plugins/calibration/sn_calibration_baseline/evaluate_extremities.py
- plugins/calibration/sn_calibration_baseline/__init__.py
- plugins/calibration/sn_calibration_baseline/soccerpitch.py
- plugins/calibration/tvcalib/cam_modules.py
- plugins/calibration/tvcalib/fuse_argmin.py
- plugins/calibration/tvcalib/fuse_stack.py
- plugins/calibration/tvcalib/inference.py
- plugins/calibration/tvcalib/module.py
- plugins/calibration/tvcalib/optimize.py
- plugins/calibration/tvcalib/README.md
- plugins/calibration/tvcalib/sncalib_dataset.py
- plugins/calibration/tvcalib/visualize_per_sample_output.ipynb
- sn_gamestate/structured_calibration/cached_primitive_eval_round3.py
- sn_gamestate/structured_calibration/__init__.py
- sn_gamestate/structured_calibration/metrics.py
- sn_gamestate/structured_calibration/primitive_fitting.py
- sn_gamestate/structured_calibration/primitive_mapping.py
- sn_gamestate/structured_calibration/primitive_weighting.py
- sn_gamestate/structured_calibration/__pycache__/cached_primitive_eval_round3.cpython-310.pyc
- sn_gamestate/structured_calibration/__pycache__/__init__.cpython-310.pyc
- sn_gamestate/structured_calibration/__pycache__/metrics.cpython-310.pyc
- sn_gamestate/structured_calibration/__pycache__/primitive_mapping.cpython-310.pyc
- sn_gamestate/structured_calibration/__pycache__/primitive_weighting.cpython-310.pyc
- sn_gamestate/structured_calibration/__pycache__/score_primitive_params_round3.cpython-310.pyc
- sn_gamestate/structured_calibration/__pycache__/weighted_solver.cpython-310.pyc
- sn_gamestate/structured_calibration/score_primitive_params_round3.py
- sn_gamestate/structured_calibration/temporal_stabilizer.py
- sn_gamestate/structured_calibration/weighted_solver.py

## Key Grep Hits


### class FramebyFrameCalib
plugins/calibration/pnlcalib/utils/utils_calib.py:98:class FramebyFrameCalib:
plugins/calibration/nbjw_calib/utils/utils_calib.py:77:class FramebyFrameCalib:
plugins/calibration/nbjw_calib/utils/utils_calib.py.bak_temporal:77:class FramebyFrameCalib:

### def heuristic_voting
plugins/calibration/pnlcalib/utils/utils_calib.py:675:    def heuristic_voting(self, refine=False, refine_lines=False):
plugins/calibration/pnlcalib/utils/utils_calib.py:696:    def heuristic_voting_ground(self, refine_lines=False):
plugins/calibration/nbjw_calib/utils/utils_calib_seq.py:495:    def heuristic_voting(self):
plugins/calibration/nbjw_calib/utils/utils_calib.py:329:    def heuristic_voting(self):
plugins/calibration/nbjw_calib/utils/utils_calib.py.bak_temporal:326:    def heuristic_voting(self):

### def get_cam_params
plugins/calibration/pnlcalib/utils/utils_calib.py:364:    def get_cam_params(self, mode='full', use_ransac=0, refine=False, refine_w_lines=False):
plugins/calibration/nbjw_calib/utils/utils_calib_seq.py:184:    def get_cam_params(self, mode='full', use_ransac=0, refine=False):
plugins/calibration/nbjw_calib/utils/utils_calib.py:204:    def get_cam_params(self, mode='full', use_ransac=0, refine=False):
plugins/calibration/nbjw_calib/utils/utils_calib.py.bak_temporal:204:    def get_cam_params(self, mode='full', use_ransac=0, refine=False):
experiments/detection_benchmark/broadtrack_min_ablation_round3.py:33:    def get_cam_params(self, mode="full", use_ransac=0, refine=False):

### calibrateCamera
plugins/calibration/pnlcalib/utils/utils_calib.py:379:            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
plugins/calibration/pnlcalib/utils/utils_calib.py:398:                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
plugins/calibration/nbjw_calib/utils/utils_calib_seq.py:336:            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_pts, self.img_pts,
plugins/calibration/nbjw_calib/utils/utils_calib.py:218:            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_pts, self.img_pts,
plugins/calibration/nbjw_calib/utils/utils_calib.py.bak_temporal:217:        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_pts, self.img_pts,
experiments/detection_benchmark/broadtrack_min_ablation_round3.py:6:- radial_k1: solve cameras with cv2.calibrateCamera allowing K1 radial distortion.
experiments/detection_benchmark/broadtrack_min_ablation_round3.py:41:            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(

### findHomography
plugins/calibration/pnlcalib/utils/utils_calib.py:201:                        h, status = cv2.findHomography(np.array(obj_list[i]), np.array(img_list[i]), cv2.RANSAC,
plugins/calibration/pnlcalib/utils/utils_calib.py:645:                H, mask = cv2.findHomography(obj_pts, img_pts, cv2.RANSAC, use_ransac)
plugins/calibration/pnlcalib/utils/utils_calib.py:647:                H, mask = cv2.findHomography(obj_pts, img_pts)
plugins/calibration/nbjw_calib/utils/utils_calib_seq.py:145:                        h, status = cv2.findHomography(np.array(obj_list[i]), np.array(img_list[i]), cv2.RANSAC, use_ransac)
plugins/calibration/nbjw_calib/utils/utils_calib_seq.py:436:            H, mask = cv2.findHomography(obj_pts, img_pts, cv2.RANSAC, use_ransac)
plugins/calibration/nbjw_calib/utils/utils_keypoints.py:352:            H, mask = cv2.findHomography(np.array(obj_points[0], dtype=np.float32), np.array(img_points[0], dtype=np.float32), cv2.RANSAC, use_ransac)
plugins/calibration/nbjw_calib/utils/utils_keypoints.py:354:            H, mask = cv2.findHomography(np.array(obj_points[0], dtype=np.float32), np.array(img_points[0], dtype=np.float32))
plugins/calibration/nbjw_calib/utils/utils_calib.py:169:                        h, status = cv2.findHomography(np.array(obj_list[i]), np.array(img_list[i]), cv2.RANSAC, use_ransac)
plugins/calibration/nbjw_calib/utils/utils_calib.py:273:            H, mask = cv2.findHomography(obj_pts, img_pts, cv2.RANSAC, use_ransac)
plugins/calibration/nbjw_calib/utils/utils_calib.py.bak_temporal:169:                        h, status = cv2.findHomography(np.array(obj_list[i]), np.array(img_list[i]), cv2.RANSAC, use_ransac)
plugins/calibration/nbjw_calib/utils/utils_calib.py.bak_temporal:270:            H, mask = cv2.findHomography(obj_pts, img_pts, cv2.RANSAC, use_ransac)
sn_gamestate/structured_calibration/weighted_solver.py:34:    H, mask = cv2.findHomography(obj, img, cv2.RANSAC, thresh)

### get_keypoints_from_heatmap
plugins/calibration/pnlcalib/utils/utils_heatmap.py:143:def get_keypoints_from_heatmap_batch_maxpool(
plugins/calibration/pnlcalib/utils/utils_heatmap.py:214:def get_keypoints_from_heatmap_batch_maxpool_l(
plugins/calibration/nbjw_calib/utils/utils_heatmap.py:143:def get_keypoints_from_heatmap_batch_maxpool(
plugins/calibration/nbjw_calib/utils/utils_heatmap.py:214:def get_keypoints_from_heatmap_batch_maxpool_l(
sn_gamestate/calibration/pnlcalib.py:21:from pnlcalib.utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool, \
sn_gamestate/calibration/pnlcalib.py:22:                                            get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, \
sn_gamestate/calibration/pnlcalib.py:124:        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
sn_gamestate/calibration/pnlcalib.py:125:        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
sn_gamestate/calibration/nbjw_calib.py:21:from nbjw_calib.utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool, \
sn_gamestate/calibration/nbjw_calib.py:22:                                            get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, \
sn_gamestate/calibration/nbjw_calib.py:122:        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
sn_gamestate/calibration/nbjw_calib.py:123:        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
experiments/detection_benchmark/smoke_inference_temporal.py:27:    get_keypoints_from_heatmap_batch_maxpool, coords_to_dict)
experiments/detection_benchmark/smoke_inference_temporal.py:36:    coords = get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1])
experiments/detection_benchmark/analyze_ransac_outlier_tracks_full49.py:23:    get_keypoints_from_heatmap_batch_maxpool,
experiments/detection_benchmark/analyze_ransac_outlier_tracks_full49.py:24:    get_keypoints_from_heatmap_batch_maxpool_l,
experiments/detection_benchmark/analyze_ransac_outlier_tracks_full49.py:38:    kc = get_keypoints_from_heatmap_batch_maxpool(kp[:, :-1])
experiments/detection_benchmark/analyze_ransac_outlier_tracks_full49.py:39:    lc = get_keypoints_from_heatmap_batch_maxpool_l(line[:, :-1])
experiments/detection_benchmark/standalone_calib_eval_full.py:25:    get_keypoints_from_heatmap_batch_maxpool,
experiments/detection_benchmark/standalone_calib_eval_full.py:26:    get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, coords_to_dict)
experiments/detection_benchmark/standalone_calib_eval_full.py:122:            kc = get_keypoints_from_heatmap_batch_maxpool(refined[:, :-1])
experiments/detection_benchmark/standalone_calib_eval_full.py:123:            lc = get_keypoints_from_heatmap_batch_maxpool_l(line_at[i].float()[:, :-1])
experiments/detection_benchmark/standalone_calib_eval.py:21:    get_keypoints_from_heatmap_batch_maxpool,
experiments/detection_benchmark/standalone_calib_eval.py:22:    get_keypoints_from_heatmap_batch_maxpool_l,
experiments/detection_benchmark/standalone_calib_eval.py:103:        kp_coords = get_keypoints_from_heatmap_batch_maxpool(refined[:, :-1])
experiments/detection_benchmark/standalone_calib_eval.py:104:        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(line[:, :-1])
experiments/detection_benchmark/fast_full_test.py:44:        get_keypoints_from_heatmap_batch_maxpool,
experiments/detection_benchmark/fast_full_test.py:45:        get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, coords_to_dict)
experiments/detection_benchmark/fast_full_test.py:80:                kc = get_keypoints_from_heatmap_batch_maxpool(refined[:, :-1])
experiments/detection_benchmark/fast_full_test.py:81:                lc = get_keypoints_from_heatmap_batch_maxpool_l(line_at[i].float()[:, :-1])
experiments/detection_benchmark/cached_outlier_only_eval_round3.py:30:    get_keypoints_from_heatmap_batch_maxpool,
experiments/detection_benchmark/cached_outlier_only_eval_round3.py:31:    get_keypoints_from_heatmap_batch_maxpool_l,
experiments/detection_benchmark/cached_outlier_only_eval_round3.py:37:    kc = get_keypoints_from_heatmap_batch_maxpool(kp_hm[:, :-1])
experiments/detection_benchmark/cached_outlier_only_eval_round3.py:38:    lc = get_keypoints_from_heatmap_batch_maxpool_l(line_hm[:, :-1])
experiments/detection_benchmark/broadtrack_min_ablation_round3.py:27:    get_keypoints_from_heatmap_batch_maxpool,
experiments/detection_benchmark/broadtrack_min_ablation_round3.py:28:    get_keypoints_from_heatmap_batch_maxpool_l,
experiments/detection_benchmark/broadtrack_min_ablation_round3.py:73:    kc = get_keypoints_from_heatmap_batch_maxpool(kp[:, :-1])
experiments/detection_benchmark/broadtrack_min_ablation_round3.py:74:    lc = get_keypoints_from_heatmap_batch_maxpool_l(line[:, :-1])
experiments/detection_benchmark/export_ransac_outlier_k5_point_cases.py:18:    get_keypoints_from_heatmap_batch_maxpool,
experiments/detection_benchmark/export_ransac_outlier_k5_point_cases.py:19:    get_keypoints_from_heatmap_batch_maxpool_l,

### model_l
sn_gamestate/temporal_hrnet/temporal_nbjw.py:148:        self.model_l = _TemporalHRNetWrapper(self.model_l, ln_a, win_ln).to(self.device)
sn_gamestate/temporal_hrnet/temporal_nbjw.py:151:            wrapper.window for wrapper in (self.model, self.model_l)
sn_gamestate/temporal_hrnet/temporal_nbjw.py:166:        self.model_l.reset()
sn_gamestate/calibration/pnlcalib.py:100:        self.model_l = get_cls_net_l(self.cfg_l)
sn_gamestate/calibration/pnlcalib.py:101:        self.model_l.load_state_dict(loaded_state_l)
sn_gamestate/calibration/pnlcalib.py:102:        self.model_l.to(device)
sn_gamestate/calibration/pnlcalib.py:103:        self.model_l.eval()
sn_gamestate/calibration/pnlcalib.py:122:            heatmaps_l = self.model_l(batch.to(self.device))
sn_gamestate/calibration/nbjw_calib.py:99:        self.model_l = get_cls_net_l(self.cfg_l)
sn_gamestate/calibration/nbjw_calib.py:100:        self.model_l.load_state_dict(loaded_state_l)
sn_gamestate/calibration/nbjw_calib.py:101:        self.model_l.to(device)
sn_gamestate/calibration/nbjw_calib.py:102:        self.model_l.eval()
sn_gamestate/calibration/nbjw_calib.py:120:            heatmaps_l = self.model_l(batch.to(self.device))

### cls_hrnet_l
sn_gamestate/calibration/pnlcalib.py:20:from pnlcalib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
sn_gamestate/calibration/nbjw_calib.py:20:from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
experiments/detection_benchmark/smoke_inference_temporal.py:25:from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
experiments/detection_benchmark/cache_hrnet_pred_only.py:18:from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
experiments/detection_benchmark/standalone_calib_eval_full.py:23:from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
experiments/detection_benchmark/fast_full_test.py:42:    from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
experiments/detection_benchmark/build_temporal_dataset.py:46:from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
experiments/detection_benchmark/eval_temporal_input_mixer_calib.py:19:from nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l

### accuracy_eval
sn_gamestate/structured_calibration/metrics.py:110:def accuracy_eval(params_by_imgid_per_video: dict, data_root, videos, nproc=10, stride=1):
experiments/detection_benchmark/standalone_calib_eval.py:4:accuracy metric (structured_calibration.metrics.accuracy_eval). For each adapter
experiments/detection_benchmark/standalone_calib_eval.py:28:from sn_gamestate.structured_calibration.metrics import accuracy_eval
experiments/detection_benchmark/standalone_calib_eval.py:139:        res = accuracy_eval(params_by_vid, DATA_ROOT, VIDS, nproc=3, stride=args.stride)

### TrackEvalEvaluator
sn_gamestate/configs/eval/gs_hota.yaml:1:_target_: tracklab.wrappers.TrackEvalEvaluator

### GS-HOTA
sn_gamestate/configs/gsr_eval_test_smoke.yaml:1:# Eval-only: load step3 smoke state, empty pipeline, auto GS-HOTA (pitch/tol5/all-attrs)
sn_gamestate/configs/gsr_eval_v10.yaml:1:# Eval-only: load step3 smoke state, empty pipeline, auto GS-HOTA (pitch/tol5/all-attrs)
sn_gamestate/configs/gsr_eval_test_15partial.yaml:1:# Eval-only: load step3 smoke state, empty pipeline, auto GS-HOTA (pitch/tol5/all-attrs)

### bbox_pitch
sn_gamestate/team/tracklet_team_side_labeling_api.py:17:    input_columns = ["track_id", "team_cluster", "bbox_pitch", "role"]
sn_gamestate/team/tracklet_team_side_labeling_api.py:31:        xa_coordinates = [bbox["x_bottom_middle"] if isinstance(bbox, dict) else np.nan for bbox in team_a.bbox_pitch]  # (x, y) are the center of a bbox
sn_gamestate/team/tracklet_team_side_labeling_api.py:32:        xb_coordinates = [bbox["x_bottom_middle"] if isinstance(bbox, dict) else np.nan for bbox in team_b.bbox_pitch]  # (x, y) are the center of a bbox
sn_gamestate/team/tracklet_team_side_labeling_api.py:45:        goalkeepers = detections[detections.role == "goalkeeper"].dropna(subset=["bbox_pitch"])
sn_gamestate/team/tracklet_team_side_labeling_api.py:46:        gk_team = goalkeepers.bbox_pitch.apply(lambda bbox: "right" if (bbox["x_bottom_middle"] > 0) else "left")
sn_gamestate/visualization/pitch.py:47:    if detections_gt is not None and "bbox_pitch" in detections_gt:
sn_gamestate/visualization/pitch.py:49:    if detections_pred is not None and "bbox_pitch" in detections_pred:
sn_gamestate/visualization/pitch.py:95:        bbox_name = "bbox_pitch"
sn_gamestate/remove_outside/remove_outside_api.py:23:    input_columns = ["track_id", "bbox_pitch"]
sn_gamestate/remove_outside/remove_outside_api.py:36:            valid_frames = group[group['bbox_pitch'].apply(lambda x: isinstance(x, dict))]
sn_gamestate/remove_outside/remove_outside_api.py:42:                bbox = frame['bbox_pitch']
sn_gamestate/calibration/bbox2pitch.py:25:    output_columns = dict(detection=["bbox_pitch"],
sn_gamestate/calibration/bbox2pitch.py:37:            detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(sn_cam))
sn_gamestate/calibration/bbox2pitch.py:39:            detections["bbox_pitch"] = detections.bbox.ltrb().apply(
sn_gamestate/calibration/bbox2pitch.py:40:                get_bbox_pitch_homography(camera_parameters)
sn_gamestate/calibration/bbox2pitch.py:44:            return pd.DataFrame(columns=["bbox_pitch"])
sn_gamestate/calibration/bbox2pitch.py:47:            return pd.DataFrame(columns=["bbox_pitch"])
sn_gamestate/calibration/bbox2pitch.py:49:        return detections["bbox_pitch"]
sn_gamestate/calibration/bbox2pitch.py:54:        for image_id, bbox_pitch in zip(metadatas.index, batch):
sn_gamestate/calibration/bbox2pitch.py:56:            image_detections["bbox_pitch"] = bbox_pitch
sn_gamestate/calibration/bbox2pitch.py:57:            output_detections.extend(image_detections["bbox_pitch"])
sn_gamestate/calibration/bbox2pitch.py:60:        return pd.DataFrame({"bbox_pitch": output_detections}, index=output_index)
sn_gamestate/calibration/bbox2pitch.py:63:def get_bbox_pitch(cam):
sn_gamestate/calibration/bbox2pitch.py:82:def get_bbox_pitch_homography(homography):
sn_gamestate/calibration/pnlcalib.py:165:        "detection": ["bbox_pitch"],
sn_gamestate/calibration/pnlcalib.py:197:                detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(h))
sn_gamestate/calibration/pnlcalib.py:204:                    detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(h))
sn_gamestate/calibration/pnlcalib.py:207:                    detections["bbox_pitch"] = None
sn_gamestate/calibration/pnlcalib.py:208:            return detections[["bbox_pitch"]], pd.DataFrame([
sn_gamestate/calibration/pnlcalib.py:214:                detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch(h))
sn_gamestate/calibration/pnlcalib.py:217:                detections["bbox_pitch"] = None
sn_gamestate/calibration/pnlcalib.py:219:            return detections[["bbox_pitch"]], pd.DataFrame([
sn_gamestate/calibration/pnlcalib.py:224:def get_bbox_pitch(h):
sn_gamestate/calibration/nbjw_calib.py:162:        "detection": ["bbox_pitch"],
sn_gamestate/calibration/nbjw_calib.py:186:                detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch_h(h))
sn_gamestate/calibration/nbjw_calib.py:193:                    detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch_h(h))
sn_gamestate/calibration/nbjw_calib.py:196:                    detections["bbox_pitch"] = None
sn_gamestate/calibration/nbjw_calib.py:197:            return detections[["bbox_pitch"]], pd.DataFrame([
sn_gamestate/calibration/nbjw_calib.py:203:                detections["bbox_pitch"] = detections.bbox.ltrb().apply(get_bbox_pitch_h(h))
sn_gamestate/calibration/nbjw_calib.py:206:                detections["bbox_pitch"] = None

### parameters
plugins/calibration/pnlcalib/utils/utils_geometry.py:106:            center, width, height, phi = reg.as_parameters()
plugins/calibration/pnlcalib/utils/utils_geometry.py:151:    # Extract ellipse parameters
plugins/calibration/pnlcalib/utils/utils_calib.py:535:        This method initializes the essential camera parameters from the homography between the world plane of the pitch
plugins/calibration/pnlcalib/utils/utils_calib.py:537:        Multiple View Geometry in computer vision, p225), then using the relation between the camera parameters and the
plugins/calibration/tvcalib/module.py:40:            self.cam_param_dict.param_dict.parameters(), lr=0.1, weight_decay=0.01
plugins/calibration/tvcalib/module.py:51:                self.cam_param_dict.param_dict_dist.parameters(), lr=1e-3, weight_decay=0.01
plugins/calibration/tvcalib/module.py:70:        # individual camera parameters & distortion parameters
plugins/calibration/tvcalib/optimize.py:135:        "camera": cam.get_parameters(batch_size),
plugins/calibration/tvcalib/sncalib_dataset.py:512:        # add camera parameters
plugins/calibration/tvcalib/README.md:4:To predict all individual camera parameters from the results of the segment localization, we can run: `python -m tvcalib.optimize`:
plugins/calibration/tvcalib/README.md:50:The folder `output_dir_prefix` contains at least the predicted camera parameters and additional information (loss, other meta infomration like stadium) for each sample (`per_sample_output.json`):
plugins/calibration/tvcalib/cam_modules.py:11:    """Holds individual camera parameters including lens distortion parameters as nn.Modul"""
plugins/calibration/tvcalib/cam_modules.py:60:        """Initializes all camera parameters with zeros and replace specific values with provided values
plugins/calibration/tvcalib/cam_modules.py:127:        As T is not provided for camra location and lens distortion, these parameters are assumed to be fixed accross T.
plugins/calibration/tvcalib/cam_modules.py:128:        phi_dict is a dict of parameters containing:
plugins/calibration/tvcalib/cam_modules.py:520:    def get_parameters(self, true_batch_size=None):
plugins/calibration/tvcalib/cam_modules.py:522:        Get dict of relevant camera parameters and homography matrix
plugins/calibration/sn_calibration_baseline/evaluate_camera.py:16:    Given a set of camera parameters, this function adapts the camera to the desired image resolution and then
plugins/calibration/sn_calibration_baseline/evaluate_camera.py:20:    :param camera_annotation: camera parameters in their json/dictionary format
plugins/calibration/sn_calibration_baseline/evaluate_camera.py:28:    cam.from_json_parameters(camera_annotation)
plugins/calibration/sn_calibration_baseline/camera.py:106:        parameters.
plugins/calibration/sn_calibration_baseline/camera.py:146:        Once that there is a minimal set of initial camera parameters (calibration, rotation and position roughly known),
plugins/calibration/sn_calibration_baseline/camera.py:162:        This method initializes the essential camera parameters from the homography between the world plane of the pitch
plugins/calibration/sn_calibration_baseline/camera.py:164:        Multiple View Geometry in computer vision, p225), then using the relation between the camera parameters and the
plugins/calibration/sn_calibration_baseline/camera.py:194:    def to_json_parameters(self):
plugins/calibration/sn_calibration_baseline/camera.py:243:    def from_json_parameters(self, calib_json_object):
plugins/calibration/sn_calibration_baseline/camera.py:245:        Loads camera parameters from dictionary.
plugins/calibration/sn_calibration_baseline/camera.py:246:        :param calib_json_object: the dictionary containing camera parameters.
plugins/calibration/sn_calibration_baseline/camera.py:309:        Uses current camera parameters to predict where a 3D point is seen by the camera.
plugins/calibration/sn_calibration_baseline/camera.py:370:        Adapts the internal parameters for image resolution changes
plugins/calibration/sn_calibration_baseline/baseline_cameras.py:136:    parser = argparse.ArgumentParser(description='Baseline for camera parameters extraction')
plugins/calibration/sn_calibration_baseline/baseline_cameras.py:246:                    camera_predictions = cam.to_json_parameters()
plugins/calibration/nbjw_calib/utils/utils_geometry.py:106:            center, width, height, phi = reg.as_parameters()
plugins/calibration/nbjw_calib/utils/utils_geometry.py:151:    # Extract ellipse parameters
plugins/calibration/nbjw_calib/utils/utils_keypoints.py:524:                                center, width, height, theta = reg.as_parameters()
plugins/calibration/nbjw_calib/utils/utils_keypoints.py:858:                                        center, width, height, theta = reg.as_parameters()
plugins/calibration/nbjw_calib/utils/utils_calib.py:292:        # Extract relevant camera parameters from the dictionary
plugins/calibration/nbjw_calib/utils/utils_calib.py.bak_temporal:289:        # Extract relevant camera parameters from the dictionary
plugins/calibration/nbjw_calib/model/transforms.py:246:        """Get the parameters for the randomized transform to be applied on image.
plugins/calibration/nbjw_calib/model/transforms.py:259:            tuple: The parameters used to apply the randomized transform

## Known Output/Cache Paths

- exists: outputs/gsr/calib_baseline_test/nbjw/states/sn-gamestate.pklz
- exists: outputs/gsr/temporal_hrnet/heatmap_cache_full_20260624
- exists: outputs/gsr/temporal_hrnet/round2_temporal_calib
- exists: outputs/gsr/temporal_hrnet/temporal_calib_results_hub
## Data Field Scan

### pklz: outputs/gsr/calib_baseline_test/nbjw/states/sn-gamestate.pklz
- zip_entries: 21
- first_entries:
  - summary.json
  - 021.pkl
  - 021_image.pkl
  - 041.pkl
  - 041_image.pkl
  - 051.pkl
  - 051_image.pkl
  - 093.pkl
  - 093_image.pkl
  - 085.pkl
- ERROR: ModuleNotFoundError: No module named 'pandas'

### heatmap cache npz
- npz_count_full_cache: 36198
- sample_npz: outputs/gsr/temporal_hrnet/heatmap_cache_full_20260624/cal2023_train/CAL23_00000/frame_0000000000.npz
  - kp_hm: shape=(58, 270, 480) dtype=float16
  - line_hm: shape=(24, 270, 480) dtype=float16
  - kp_gt: shape=(58, 270, 480) dtype=float16
  - kp_mask: shape=(58,) dtype=float16
  - line_gt: shape=(24, 270, 480) dtype=float16
  - frame: shape=() dtype=int64
- sample_npz: outputs/gsr/temporal_hrnet/heatmap_cache_full_20260624/cal2023_train/CAL23_00001/frame_0000000001.npz
  - kp_hm: shape=(58, 270, 480) dtype=float16
  - line_hm: shape=(24, 270, 480) dtype=float16
  - kp_gt: shape=(58, 270, 480) dtype=float16
  - kp_mask: shape=(58,) dtype=float16
  - line_gt: shape=(24, 270, 480) dtype=float16
  - frame: shape=() dtype=int64
- sample_npz: outputs/gsr/temporal_hrnet/heatmap_cache_full_20260624/cal2023_train/CAL23_00002/frame_0000000002.npz
  - kp_hm: shape=(58, 270, 480) dtype=float16
  - line_hm: shape=(24, 270, 480) dtype=float16
  - kp_gt: shape=(58, 270, 480) dtype=float16
  - kp_mask: shape=(58,) dtype=float16
  - line_gt: shape=(24, 270, 480) dtype=float16
  - frame: shape=() dtype=int64

### existing result jsons
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/debug_camera_132/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/kalman_camera_baseline_full49/eval_full49/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/kalman_keypoint_baseline_full49/eval_full49/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/residual_scale_rs01_full/eval_full49/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/residual_scale_rs01_full_stgcn_200/eval_full49/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/residual_scale_screen/eval_videos116_123/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round2_cached_final_only/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round3_broadtrack_10vid_s20/eval/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round3_broadtrack_flow_window_116117_s20/eval/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round3_broadtrack_flow_window_116_s20/eval/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round3_broadtrack_min_noflow2_stride100/eval/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round3_broadtrack_min_radial8_stride100/eval/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round3_outlier_only_full49_gpu7/eval_full49/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round3_outlier_only_gshota3_stride1_gpu7/eval_stride1/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round3_outlier_only_subset_gpu7/eval_subset/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round3_temporal_k_sweep/eval_subset/result.json
- outputs/gsr/temporal_hrnet/temporal_calib_results_hub/round3_temporal_lr_smoke/eval_subset/result.json
## pklz Field Scan With Project Python

- pklz: outputs/gsr/calib_baseline_test/nbjw/states/sn-gamestate.pklz
- zip_entries: 21
  - summary.json
  - 021.pkl
  - 021_image.pkl
  - 041.pkl
  - 041_image.pkl
  - 051.pkl
  - 051_image.pkl
  - 093.pkl
  - 093_image.pkl
  - 085.pkl
  - 085_image.pkl
  - 034.pkl

### sample 021.pkl
- type: DataFrame
- shape: (12113, 28)
- columns:
  - role
  - jersey_number
  - jersey_number_confidence
  - body_masks
  - track_bbox_pred_kf_ltwh
  - matched_with
  - hits
  - state
  - bbox_pitch
  - category_id
  - age
  - visibility_scores
  - legibility_score
  - costs
  - role_detection
  - bbox_conf
  - time_since_update
  - track_id
  - team
  - role_confidence
  - embeddings
  - image_id
  - bbox_ltwh
  - track_bbox_kf_ltwh
  - ignored
  - team_cluster
  - video_id
  - jersey_number_detection
- first_row_non_null_fields:
  - role: str
  - jersey_number_confidence: float64
  - body_masks: ndarray
  - track_bbox_pred_kf_ltwh: ndarray
  - matched_with: float64
  - hits: int64
  - state: str
  - bbox_pitch: dict
    keys: ['x_bottom_left', 'x_bottom_middle', 'x_bottom_right', 'y_bottom_left', 'y_bottom_middle', 'y_bottom_right']
  - category_id: int64
  - age: int64
  - visibility_scores: ndarray
  - legibility_score: float64
  - costs: float64
  - role_detection: str
  - bbox_conf: float64
  - time_since_update: int64
  - track_id: float64
  - team: str
  - role_confidence: float64
  - embeddings: ndarray
  - image_id: str
  - bbox_ltwh: ndarray
  - track_bbox_kf_ltwh: ndarray
  - ignored: bool
  - team_cluster: float64
  - video_id: str
  - jersey_number_detection: float64
