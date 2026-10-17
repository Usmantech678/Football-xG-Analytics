
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

df = pd.read_csv("data/shots.csv")

goal_col = next(
    (c for c in ["goal", "is_goal", "shot_goal"] if c in df.columns),
    None
)
if goal_col is None:
    raise ValueError(f"Goal column not found. Columns: {list(df.columns)}")

df[goal_col] = pd.to_numeric(df[goal_col], errors="coerce")
df = df.dropna(subset=["x", "y", goal_col, "match_id"]).copy()
df[goal_col] = df[goal_col].astype(int)

dx = 120 - df["x"]
dy = 40 - df["y"]
df["distance"] = np.sqrt(dx**2 + dy**2)
df["angle"] = np.arctan2(
    abs(8 * dx),
    dx**2 + (36 - df["y"]) * (44 - df["y"])
)

numeric = ["x", "y", "distance", "angle"]
categorical = [
    c for c in
    ["shot_type", "body_part", "technique", "under_pressure"]
    if c in df.columns
]
features = numeric + categorical

for col in categorical:
    df[col] = df[col].fillna("Unknown").astype(str)

X = df[features]
y = df[goal_col]
groups = df["match_id"]

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

def make_preprocessor(scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

models = {
    "Logistic Regression": Pipeline([
        ("prep", make_preprocessor(True)),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest": Pipeline([
        ("prep", make_preprocessor(False)),
        ("model", RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

predictions = {}

print("Training shots:", len(train_idx))
print("Testing shots:", len(test_idx))
print("Test matches:", groups.iloc[test_idx].nunique())

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions[name] = probabilities

    print(name)
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    print("Log loss:", round(log_loss(y_test, probabilities), 4))
    print("Brier score:", round(brier_score_loss(y_test, probabilities), 4))

plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")

for name, probabilities in predictions.items():
    actual, predicted = calibration_curve(
        y_test, probabilities, n_bins=10, strategy="quantile"
    )
    plt.plot(predicted, actual, marker="o", label=name)

plt.xlabel("Mean predicted probability")
plt.ylabel("Actual goal frequency")
plt.title("xG Model Calibration — Held-out Matches")
plt.legend()
plt.tight_layout()
plt.savefig("calibration.png", dpi=150)
plt.show()
