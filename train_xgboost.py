
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

df = pd.read_csv("data/shots_with_context.csv")

dx = 120 - df["x"]
dy = 40 - df["y"]
df["distance"] = np.sqrt(dx**2 + dy**2)
df["angle"] = np.arctan2(
    abs(8 * dx),
    dx**2 + (36 - df["y"]) * (44 - df["y"])
)

numeric = [
    "x", "y", "distance", "angle",
    "goalkeeper_distance", "goalkeeper_goal_distance",
    "nearest_defender_distance", "defenders_within_5m",
    "defenders_in_shot_lane", "visible_defender_count",
    "visible_teammate_count"
]
categorical = [
    "shot_type", "body_part", "technique", "under_pressure"
]

df = df.dropna(subset=["match_id", "goal"])
X = df[numeric + categorical]
y = df["goal"].astype(int)
groups = df["match_id"]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

evaluation_preprocessor = preprocessor
X_train_encoded = evaluation_preprocessor.fit_transform(X_train)
X_test_encoded = evaluation_preprocessor.transform(X_test)

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5,
    min_child_weight=5,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost...")
model.fit(X_train_encoded, y_train)

probabilities = model.predict_proba(X_test_encoded)[:, 1]

print("\nHeld-out evaluation:")
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("Log loss:", round(log_loss(y_test, probabilities), 4))
print("Brier score:", round(brier_score_loss(y_test, probabilities), 4))

print("\nTraining final model on all shots...")
X_all_encoded = preprocessor.fit_transform(X)
model.fit(X_all_encoded, y)

joblib.dump(
    {"preprocessor": preprocessor, "model": model, "features": numeric + categorical},
    "xgboost_model.joblib"
)

print("Saved xgboost_model.joblib")
