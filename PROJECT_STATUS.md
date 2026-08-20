# SoccerMaster Project Status

Updated: 2026-06-03T14:05:25

## Goal
Run the full SoccerMaster smoke pipeline on server data `valid/SNGS-021` under `/remote-home/jiayuanrao/yishan/SoccerMaster`.

## Confirmed
- Conda env: `/remote-home/jiayuanrao/tools/anaconda/anaconda3/envs/wys_soccermaster/bin/python`
- PyTorch CUDA: available, 8 GPUs visible
- Data: `/remote-home/jiayuanrao/yishan/sn-gamestate/data/SoccerNetGS/valid/SNGS-021/img1`, 750 frames
- Smoke configs created:
  - `codes/sn-gamestate/sn_gamestate/configs/gsr_step_1_valid_021.yaml`
  - `codes/sn-gamestate/sn_gamestate/configs/gsr_step_3_valid_021_accelerate.yaml`
  - `codes/sam2/step_2/gsr_step2_valid_021.sh`
  - `codes/sam2/step_2/merge_valid_021.sh`
- Entrypoint updated: `run.sh`

## Links/paths prepared
- `codes/sn-gamestate/datasets/SoccerNetGS` -> existing SoccerNetGS data
- `codes/sn-gamestate/pretrained_models` -> `codes/SoccerMaster/pretrained_models`
- `codes/SoccerMaster/pretrained_models/jn` -> root `pretrained_models/jn`
- YOLO compatibility link created if needed: `yolo_v8x6_person_lr_default_best.pt` -> `yolo_v8x6_finetuned.pt`

## Current status
Patched configs/scripts. Next: run Step 1 and inspect failures.

## 2026-06-03T14:10:07
- Step 1 failed on eager PoseTrack eval import via `tracklab.wrappers.__init__`. Removed top-level `from .eval import *`; eval modules remain importable from `tracklab.wrappers.eval`.

## 2026-06-03T14:15:30
- Narrowed `tracklab.wrappers.__init__` to dataset wrappers only to avoid optional tracker/eval dependency failures during Hydra lookup.

## 2026-06-03T14:16:53
- Added direct top-level export for `TrackEvalEvaluator` without importing optional PoseTrack eval package.

## 2026-06-03T14:18:28
- Narrowed `tracklab.wrappers.eval.__init__` to `trackeval_evaluator` only; avoids broken optional `posetrack21` import.

## 2026-06-03T14:20:08
- Added direct top-level export for `BPBReIDStrongSORT`; avoided importing other tracker wrappers that require missing `strong_sort.deep.models`.

## 2026-06-03T14:21:18
- Narrowed `tracklab.wrappers.track.__init__` to `BPBReIDStrongSORT` only; avoids missing `strong_sort.deep.models`.

## 2026-06-03T14:22:42
- ReID failed because `reid/0` already existed. Updated `run.sh` to set unique `SLURM_JOBID` for Step 1 and Step 3.

## 2026-06-03T14:23:59
- ReID `project.job_id` requires an int; changed `run.sh` SLURM_JOBID values to plain Unix timestamps.

## 2026-06-03T14:28:42
- Step 1 succeeded: `codes/sn-gamestate/outputs/gsr/step_1_valid_021/states/sn-gamestate.pklz` (149M).
- Searching for SAM2 checkpoint before Step 2.

## 2026-06-03T15:00:23
- SAM2 checkpoint downloaded: `codes/sam2/checkpoints/sam2.1_hiera_large.pt` (857M).
- Step 2 inference succeeded: `codes/sam2/outputs/gsr_step2_valid_021/video_results/021_result.pkl` (2.8M).

## 2026-06-03T15:47:05
- Step 3 still running during monitor check; process active on GPU 6, no fatal error in log yet.

## 2026-06-03T15:47:46
- Full smoke pipeline completed for `valid/SNGS-021`.
- Step 1 output: `codes/sn-gamestate/outputs/gsr/step_1_valid_021/states/sn-gamestate.pklz` (~149M).
- Step 2 inference output: `codes/sam2/outputs/gsr_step2_valid_021/video_results/021_result.pkl` (~2.8M).
- Step 2 merge output: `codes/sam2/outputs/gsr_step2_valid_021/refined_sn-gamestate.pklz` (~155M).
- Step 3 outputs:
  - `codes/sn-gamestate/outputs/gsr/step_3_valid_021/states/sn-gamestate.pklz` (~165M)
  - `codes/sn-gamestate/outputs/gsr/step_3_valid_021/visualization/videos/021.mp4` (~102M)
- Repro command: `cd /remote-home/jiayuanrao/yishan/SoccerMaster && bash run.sh`.
- Notes: only `valid` split exists in linked SoccerNetGS data, so train/test/challenge warnings are expected for this smoke run.

## 2026-06-03T15:59:52
- Created eval-only config `codes/sn-gamestate/sn_gamestate/configs/gsr_eval_valid_021.yaml` loading final Step 3 state.

## 2026-06-03T16:02:05
- Evaluation completed with `EVAL_RC=0` using `gsr_eval_valid_021.yaml`.
- Loaded final state: `codes/sn-gamestate/outputs/gsr/step_3_valid_021/states/sn-gamestate.pklz`.
- Eval outputs:
  - predictions: `codes/sn-gamestate/outputs/gsr/eval_valid_021/eval/pred/SoccerNetGS-valid/tracklab/SNGS-021.json`
  - zipped predictions: `codes/sn-gamestate/outputs/gsr/eval_valid_021/eval/pred/SoccerNetGS-valid.zip`
  - summaries: `codes/sn-gamestate/outputs/gsr/eval_valid_021/eval/results/tracklab/`
- Combined metrics for `valid/SNGS-021`:
  - GS-HOTA/HOTA: 59.501
  - DetA: 49.595
  - AssA: 71.400
  - MOTA: 33.521
  - MOTP: 93.118
  - IDF1: 65.027
  - IDR: 68.228
  - IDP: 62.113
  - Dets / GT_Dets: 13065 / 11894
  - IDs / GT_IDs: 35 / 23

## 2026-06-03T16:14:20
- Added replacement experiment adapters under `experiments/replacement_adapters/`: SAM2-IoU tracker state builder and SAM3 gated-model availability checker.

## 2026-06-03T16:17:10
- Updated SAM2-IoU adapter to support SAM2 refined pklz files without `summary.json`.

## 2026-06-03T16:19:28
- Added Step3/eval configs for experiment B (`sam2_iou_tracker`).

## 2026-06-03T16:22:04
- Fixed SAM2-IoU adapter to write `summary.columns` for TrackLab `TrackerState` loading.

## 2026-06-04 Detection-only SAM3 benchmark v1
- Added detection-only runner: `experiments/detection_benchmark/detection_benchmark.py`.
- Fixed sample manifest: `experiments/detection_benchmark/manifests/valid_seed20260604_10.json` with videos `SNGS-021,023,034,040,041,051,052,085,091,093`.
- Smoke run (`SNGS-021`, 20 frames) completed under `experiments/detection_benchmark/runs/smoke_021_20f/`.
- Full SAM3 10-video inference completed under `experiments/detection_benchmark/runs/valid10_seed20260604/`.
- Full SAM3 image-space report: `experiments/detection_benchmark/runs/valid10_seed20260604/REPORT.md`.
- Main 10-video aggregate results:
  - Best Stage-1 prompt: `player`.
  - `best_nms_geom_field`: AP50 0.7731, mAP 0.4091, image_DetA 0.7386, image_DetPr 0.8610, image_DetRe 0.8377, pred/gt 115692/119461.
  - `sam3_text_player`: AP50 0.7254, mAP 0.3825, image_DetA 0.6986.
  - `sam3_text_soccer_player`: AP50 0.7214, mAP 0.3802, image_DetA 0.7012.
  - `sam3_text_football_player`: zero detections on these 10 videos.
- Existing project control states found only for `SNGS-021`, not the 10-video sample.
  - `control_021_soccermaster_full`: AP50 0.7907, mAP 0.5111, image_DetA 0.7747.
  - `control_021_sam2_iou`: AP50 0.7907, mAP 0.5100, image_DetA 0.7703.
- Official GS-HOTA detection submetrics remain auxiliary only; pure SAM3 detection-only groups do not produce reliable `bbox_pitch` without calibration.
- `best_nms_geom_field_prevbox` currently reuses text detections because installed Transformers SAM3 video API support for bbox prompting is not implemented in this runner yet; marked in report notes.

## SAM3.1 detection-only smoke - 2026-06-04
- Runner: `experiments/detection_benchmark/sam31_detection_benchmark.py`
- Output: `experiments/detection_benchmark/runs/sam31_smoke_021_20f/`
- Compatibility patch: `codes/sam3_official/sam3/model/sam3_multiplex_base.py` changed one PyTorch-2.4 compatibility line: bool `argsort` -> int32 `argsort`.
- Smoke data: `valid/SNGS-021`, first 20 frames, detection-only bbox image-space eval.
- Successful groups:
  - `sam31_text_player`: AP50 0.9644, mAP 0.7289, image_DetA 0.8765, 338/300 pred/gt, 4.839 FPS.
  - `sam31_text_player_nms_geom`: same metrics on this 20-frame smoke because raw boxes already passed filters.
  - `sam31_text_player_nms_geom_conf015`: same metrics on this smoke.
  - `sam31_text_player_nms_geom_field`: currently same as geom/conf smoke; field-specific filtering still needs full implementation.
  - `sam31_text_player_prevbox`: AP50 0.9199, mAP 0.6848, image_DetA 0.8765, 338/300 pred/gt, 6.696 FPS excluding cached text pass.
- Failed/blocked group:
  - `sam31_text_player_prevbox_chunk`: official SAM3.1 `propagate_in_video` hit tracker memory shape error (`Tensor sizes: [5184, 0, 256]`); recorded as failed group, not used as success metric.
- Visualizations: `experiments/detection_benchmark/runs/sam31_smoke_021_20f/visualization/<group>/vis.mp4`
