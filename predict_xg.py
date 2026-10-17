
import joblib
import pandas as pd

model = joblib.load("xg_model.joblib")

x = float(input("Shot X coordinate (0-120): "))
y = float(input("Shot Y coordinate (0-80): "))
minute = float(input("Match minute: "))
shot_type = input("Shot type (e.g. Open Play): ")
body_part = input("Body part (e.g. Right Foot): ")
technique = input("Technique (e.g. Normal): ")
pressure = input("Under pressure? (yes/no): ").strip().lower()

dx = 120 - x
dy = 40 - y
distance = (dx**2 + dy**2) ** 0.5

lower = 36 - y
upper = 44 - y
import math
angle = math.atan2(abs(8 * dx), dx**2 + lower * upper)

shot = pd.DataFrame([{
    "x": x,
    "y": y,
    "minute": minute,
    "distance": distance,
    "angle": angle,
    "shot_type": shot_type,
    "body_part": body_part,
    "technique": technique,
    "under_pressure": str(pressure in ["yes", "y", "true"])
}])

probability = model.predict_proba(shot)[0][1]

print(f"\nPredicted xG: {probability:.1%}")
