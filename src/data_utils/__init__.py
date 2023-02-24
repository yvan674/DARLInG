from pathlib import Path

# Maps which rooms were used on which dates
ROOM_DATE_MAPPING = {
    1: [Path(fp) for fp in [
        "20181109",
        "20181112",
        "20181115",
        "20181116",
        "20181118",
        "20181121",
        "20181127",
        "20181128",
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
