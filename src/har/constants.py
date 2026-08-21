from __future__ import annotations

from types import MappingProxyType

ACTIVITY_CODES = tuple("ABCDEFGHIJKLMOPQRS")  # no N
LABEL_ORDER = ACTIVITY_CODES

CODE_TO_NAME = MappingProxyType(
    {
        "A": "walking",
        "B": "jogging",
        "C": "stairs",
        "D": "sitting",
        "E": "standing",
        "F": "typing",
        "G": "brushing teeth",
        "H": "eating soup",
        "I": "eating chips",
        "J": "eating pasta",
        "K": "drinking",
        "L": "eating sandwich",
        "M": "kicking",
        "O": "catch",
        "P": "dribbling",
        "Q": "writing",
        "R": "clapping",
        "S": "folding clothes",
    }
)

GROUP_OF = MappingProxyType(
    {
        "A": "locomotion",
        "B": "locomotion",
        "C": "locomotion",
        "M": "locomotion",
        "D": "posture",
        "E": "posture",
        "F": "hand",
        "G": "hand",
        "O": "hand",
        "P": "hand",
        "Q": "hand",
        "R": "hand",
        "S": "hand",
        "H": "eating",
        "I": "eating",
        "J": "eating",
        "K": "eating",
        "L": "eating",
    }
)

TARGET_HZ = 20.0
DEFAULT_WINDOW_S = 5.0
DEFAULT_HOP_S = 1.0
DEFAULT_TRIM_START_S = 0.0  # set 15.0 in an ablation config
SUBJECT_ID_MIN, SUBJECT_ID_MAX = 1600, 1650
UCI_PAGE = (
    "https://archive.ics.uci.edu/dataset/507/"
    "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset"
)
