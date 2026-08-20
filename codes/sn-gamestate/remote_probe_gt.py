from experiments.detection_benchmark import eval_temporal_input_mixer_calib as base

gt = base.load_gt_lines_for_video(base.DATA_ROOT, "116")
print(type(gt), len(gt))
for key, value in list(gt.items())[:1]:
    print("key", key)
    print("value_type", type(value))
    print("value_repr", repr(value)[:3000])
