"""Entirely fictional runner histories for offline LightGBM smoke tests."""
from datetime import date, timedelta

from .features import DATA_FIELDS


def synthetic_rows(days=48):
    rows = []
    for day_index in range(days):
        day = date(2020, 1, 1) + timedelta(days=day_index * 8)
        winner = (day_index // 8) % 8
        for horse in range(8):
            row = {key: "" for key in DATA_FIELDS}
            row.update(kaisai_nen=str(day.year), kaisai_tsukihi=day.strftime("%m%d"),
                       keibajo_code="05", kaisai_kai="01", kaisai_nichime="01", race_bango="01",
                       horse_number=f"{horse + 1:02d}", horse_id=f"SYNTHETIC_HORSE_{horse + 1:02d}",
                       finish_position=str((horse - winner) % 8 + 1),
                       surface="SYNTHETIC_TURF", going="SYNTHETIC_GOOD", distance="1600",
                       declared_runners="8", frame=str(horse + 1), age="4",
                       sex="SYNTHETIC_F", weight_kg="55", finish_time_seconds=str(95 + horse),
                       start_time="1500")
            rows.append(row)
    return rows
