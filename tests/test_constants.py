from pathlib import Path

from har.constants import (
    ACTIVITY_CODES,
    CHANNEL_NAMES,
    CODE_TO_NAME,
    GROUP_OF,
    LABEL_ORDER,
    TARGET_HZ,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_KEY = REPO_ROOT / "data" / "activity_key.txt"

KNOWN_GROUPS = frozenset({"locomotion", "posture", "hand", "eating"})


def test_eighteen_activity_codes_and_no_n():
    assert len(ACTIVITY_CODES) == 18
    assert "N" not in ACTIVITY_CODES
    assert ACTIVITY_CODES == tuple("ABCDEFGHIJKLMOPQRS")


def test_label_order_matches_activity_codes():
    assert LABEL_ORDER == ACTIVITY_CODES


def test_every_code_has_a_group():
    assert set(GROUP_OF) == set(ACTIVITY_CODES)
    assert set(GROUP_OF.values()) <= KNOWN_GROUPS


def test_code_to_name_has_eighteen_entries():
    assert len(CODE_TO_NAME) == 18
    assert set(CODE_TO_NAME) == set(ACTIVITY_CODES)


def test_code_to_name_matches_activity_key_file():
    from_file = {}
    for raw in ACTIVITY_KEY.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        name, code = (part.strip() for part in line.split("=", 1))
        from_file[code] = name
    assert dict(CODE_TO_NAME) == from_file


def test_target_hz_is_20():
    assert TARGET_HZ == 20.0


def test_channel_names_are_accel_then_gyro():
    assert CHANNEL_NAMES == ("ax", "ay", "az", "gx", "gy", "gz")
