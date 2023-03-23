from pathlib import Path
from data_utils.widar_dataset import WidarDataset

# Maps which rooms were used on which dates
ROOM_DATE_MAPPING = {
    1: [Path(fp) for fp in [
        "20181109",
        "20181112",
        "20181115",
        "20181116",
        "20181121",
        "20181130"
    ]],
    2: [Path(fp) for fp in [
        "20181117",
        "20181118",
        "20181127",
        "20181128",
        "20181204",
        "20181205",
        "20181208",
        "20181209",
    ]],
    3: [Path(fp) for fp in [
        "20181211"
    ]]
}

DATE_ROOM_MAPPING = {
    "20181109": 1,
    "20181112": 1,
    "20181115": 1,
    "20181116": 1,
    "20181121": 1,
    "20181130": 1,
    "20181117": 2,
    "20181118": 2,
    "20181127": 2,
    "20181128": 2,
    "20181204": 2,
    "20181205": 2,
    "20181208": 2,
    "20181209": 2,
    "20181211": 3
}

# All of these are sets, since checking set membership is about 28% faster with
# sets instead of tuples and around 30% faster than with lists.
TRAINING_SELECTION = {
    "user": {"1", "2", "4", "5"},
    "room_num": {1, 2},
    "torso_location": {"1", "2", "3", "4", "5"},
}

VALIDATION_SELECTION = {
    "user": {"10", "11", "12", "13", "14", "15", "16", "17"},
    "room_num": {1},
    "torso_location": {"1", "2", "3", "4", "5"}
}

TEST_ROOM_SELECTION = {
    "user": {"3", "7", "8", "9"},
    "room_num": {3},
    "torso_location": {"1", "2", "3", "4", "5"}
}

TEST_LOCATION_SELECTION = {
    "user": {"1"},
    "room_num": {1},
    "torso_location": {"6", "7", "8"}
}


def calculate_total_dirs(csi_dir: Path) -> int:
    """Calculates total dirs to traverse for use by a progress bar."""
    total_dirs = 0
    for date_dir in csi_dir.iterdir():
        if date_dir.is_dir():
            for user_dir in date_dir.iterdir():
                if user_dir.is_dir():
                    total_dirs += 1
    return total_dirs


__all__ = ["WidarDataset", "calculate_total_dirs", "ROOM_DATE_MAPPING",
           "DATE_ROOM_MAPPING", "TRAINING_SELECTION", "VALIDATION_SELECTION",
           "TEST_LOCATION_SELECTION", "TEST_ROOM_SELECTION"]