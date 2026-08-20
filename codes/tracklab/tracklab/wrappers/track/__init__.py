# SoccerNet smoke run only needs BPBReIDStrongSORT.
# Avoid importing optional tracker backends with incomplete dependencies.
from .bpbreid_strong_sort_api import BPBReIDStrongSORT
