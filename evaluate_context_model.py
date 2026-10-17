
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/shots_with_context.csv")

df = df.dropna(subset=["goal", "match_id", "x", "y"]).copy()
df["goal"] = df["goal"].astype(int)

dx = 120 - df["x"]
dy = 40 - df["y"]
df["distance"] = np.sqrt(dx**2 + dy**2)
df["angle"] = np.arctan2(
    abs(8 * dx),
    dx**2 + (36 - df["y"]) * (44 - df["y"])
)

numeric_base = ["x", "y", "distance", "angle"]
categorical = [
    "shot_type",
    "body_part",
    "technique",
    "under_pressure",
]

context_features = [
    "goalkeeper_distance",
    "goalkeeper_goal_distance",
    "nearest_defender_distance",
    "defenders_within_5m",
    "defenders_in_shot_lane",
    "visible_defender_count",
    "visible_teammate_count",
]

for col in categorical:
    df[col] = df[col].fillna("Unknown").astype(str)

for col in numeric_base + context_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(df, df["goal"], groups=df["match_id"])
)

train = df.iloc[train_idx]
test = df.iloc[test_idx]

print("Training shots:", len(train))
print("Testing shots:", len(test))
print("Test matches:", test["match_id"].nunique())


def build_model(numeric_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", LogisticRegression(max_iter=2000)),
    ])


models = {
    "Baseline Logistic Regression": build_model(numeric_base),
    "Context-Enhanced Logistic Regression": build_model(
        numeric_base + context_features
    ),
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(train, train["goal"])
    probabilities = model.predict_proba(test)[:, 1]

    auc = roc_auc_score(test["goal"], probabilities)
    loss = log_loss(test["goal"], probabilities, labels=[0, 1])
    brier = brier_score_loss(test["goal"], probabilities)

    results.append({
        "Model": name,
        "ROC-AUC": auc,
        "Log loss": loss,
        "Brier score": brier,
    })

    print(f"ROC-AUC: {auc:.4f}")
    print(f"Log loss: {loss:.4f}")
    print(f"Brier score: {brier:.4f}")

print("\n--- FINAL COMPARISON ---")
print(pd.DataFrame(results).to_string(index=False))
