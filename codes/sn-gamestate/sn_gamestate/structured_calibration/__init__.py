"""structured_calibration: pluggable post-processing for soccer field calibration.

Two complementary modules sit between raw per-frame calibration (NBJW/PnL/SoccerMaster)
and BEV projection / GSR evaluation:

  1. Hierarchical Field Graph (spatial)  -- TODO (staged after temporal PoC)
  2. Structure-aware Temporal Pose Stabilizer (temporal) -- temporal_stabilizer.py

Everything operates on the camera `parameters` dict already stored per-frame in the
TrackLab pklz state, so it requires no GPU re-run and no heatmaps/masks for the v1.
"""
