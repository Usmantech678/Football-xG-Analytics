
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
EVENTS_DIR = DATA_DIR / "events"
SHOTS_FILE = DATA_DIR / "shots.csv"
OUTPUT_FILE = DATA_DIR / "shots_with_context.csv"

shots = pd.read_csv(SHOTS_FILE)


def distance(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def distance_to_segment(point, start, end):
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    point = np.array(point, dtype=float)

    segment = end - start
    length_squared = np.dot(segment, segment)

    if length_squared == 0:
        return distance(point, start), 0.0

    t = np.dot(point - start, segment) / length_squared
    t = np.clip(t, 0.0, 1.0)
    closest = start + t * segment

    return float(np.linalg.norm(point - closest)), float(t)


records = []
files = list(EVENTS_DIR.glob("*.json"))

for file_index, file_path in enumerate(files, start=1):
    match_id = int(file_path.stem)

    with open(file_path, encoding="utf-8") as f:
        events = json.load(f)

    for event in events:
        if event.get("type", {}).get("name") != "Shot":
            continue

        location = event.get("location")
        if not location or len(location) < 2:
            continue

        shot = event.get("shot", {})
        freeze_frame = shot.get("freeze_frame") or []

        sx, sy = float(location[0]), float(location[1])

        # Use the shot's actual end location to identify the target goal.
        end_location = shot.get("end_location")
        if end_location and len(end_location) >= 2:
            goal_x = 120.0 if float(end_location[0]) >= 60 else 0.0
        else:
            goal_x = 120.0 if sx < 60 else 0.0

        goal_center = [goal_x, 40.0]
        start = [sx, sy]

        goalkeeper = None
        defenders = []
        teammates = []

        for player in freeze_frame:
            player_location = player.get("location")
            if not player_location or len(player_location) < 2:
                continue

            px, py = map(float, player_location[:2])
            position = player.get("position", {}).get("name", "")
            is_teammate = player.get("teammate")

            if position == "Goalkeeper" and is_teammate is False:
                goalkeeper = [px, py]
            elif is_teammate is False:
                defenders.append([px, py])
            elif is_teammate is True:
                teammates.append([px, py])

        if goalkeeper is not None:
            goalkeeper_distance = distance(start, goalkeeper)
            goalkeeper_goal_distance = distance(goalkeeper, goal_center)
        else:
            goalkeeper_distance = np.nan
            goalkeeper_goal_distance = np.nan

        defender_distances = [
            distance(start, defender) for defender in defenders
        ]

        nearest_defender = (
            min(defender_distances) if defender_distances else np.nan
        )

        defenders_within_5m = sum(
            d <= 5.0 for d in defender_distances
        )

        defenders_in_lane = 0

        for defender in defenders:
            lane_distance, t = distance_to_segment(
                defender, start, goal_center
            )

            if 0.0 < t < 1.0 and lane_distance <= 2.5:
                defenders_in_lane += 1

        records.append({
            "match_id": match_id,
            "team": event.get("team", {}).get("name", ""),
            "player": event.get("player", {}).get("name", ""),
            "minute": event.get("minute"),
            "x": round(sx, 1),
            "y": round(sy, 1),
            "goalkeeper_distance": goalkeeper_distance,
            "goalkeeper_goal_distance": goalkeeper_goal_distance,
            "nearest_defender_distance": nearest_defender,
            "defenders_within_5m": defenders_within_5m,
            "defenders_in_shot_lane": defenders_in_lane,
            "visible_defender_count": len(defenders),
            "visible_teammate_count": len(teammates),
            "has_goalkeeper_data": int(goalkeeper is not None),
            "has_freeze_frame": int(bool(freeze_frame)),
        })

    if file_index % 50 == 0:
        print(f"Processed {file_index}/{len(files)} match files")

context = pd.DataFrame(records)

# Match extracted event features to the original shot rows.
# Occurrence counters handle repeated shots sharing the same key.
keys = ["match_id", "team", "player", "minute", "x", "y"]

for frame in (shots, context):
    frame["team"] = frame["team"].fillna("").astype(str)
    frame["player"] = frame["player"].fillna("").astype(str)
    frame["minute"] = pd.to_numeric(frame["minute"], errors="coerce")
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce").round(1)
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce").round(1)

shots["_match_occurrence"] = shots.groupby(
    keys, dropna=False
).cumcount()

context["_match_occurrence"] = context.groupby(
    keys, dropna=False
).cumcount()

merged = shots.merge(
    context,
    on=keys + ["_match_occurrence"],
    how="left",
    validate="one_to_one",
)

merged.drop(columns=["_match_occurrence"], inplace=True)
merged.to_csv(OUTPUT_FILE, index=False)

print("\n--- Feature extraction summary ---")
print("Original shots:", len(shots))
print("Extracted shot events:", len(context))
print("Shots matched:", int(merged["has_freeze_frame"].notna().sum()))
print(
    "Matched with goalkeeper:",
    int(merged["has_goalkeeper_data"].fillna(0).sum()),
)
print(
    "Average defenders in shot lane:",
    round(merged["defenders_in_shot_lane"].mean(), 3),
)
print("Saved:", OUTPUT_FILE)
print("\nNew feature columns:")
print([
    "goalkeeper_distance",
    "goalkeeper_goal_distance",
    "nearest_defender_distance",
    "defenders_within_5m",
    "defenders_in_shot_lane",
    "visible_defender_count",
    "visible_teammate_count",
])
