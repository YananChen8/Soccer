# SoccerNet smoke run: keep top-level Hydra symbols minimal.
# Avoid eager imports of optional PoseTrack and tracker backends with fragile dependencies.
from .dataset import *
from .eval.trackeval_evaluator import TrackEvalEvaluator
from .track.bpbreid_strong_sort_api import BPBReIDStrongSORT
