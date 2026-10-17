
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

df = pd.read_csv("data/shots.csv")

dx = 120 - df["x"]
dy = 40 - df["y"]

df["distance"] = np.sqrt(dx**2 + dy**2)
df["angle"] = np.arctan2(
    abs(8 * dx),
    dx**2 + (36 - df["y"]) * (44 - df["y"])
)

numeric = ["x", "y", "distance", "angle"]
categorical = [
    "shot_type", "body_part", "technique", "under_pressure"
]

for col in categorical:
    df[col] = df[col].fillna("Unknown").astype(str)

df = df.dropna(subset=numeric + ["goal", "match_id"])

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.2, random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, df["goal"], groups=df["match_id"])
)

train = df.iloc[train_idx]
test = df.iloc[test_idx]

print("Training matches:", train["match_id"].nunique())
print("Testing matches:", test["match_id"].nunique())
print("Training shots:", len(train))
print("Testing shots:", len(test))

def build_model(num_features, cat_features):
    transformers = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num_features)
    ]

    if cat_features:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat_features
        ))

    pipeline = Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("model", LogisticRegression(max_iter=1000))
    ])

    return pipeline

experiments = {
    "Distance-only baseline": (["distance"], []),
    "Full feature model": (numeric, categorical)
}

for name, (nums, cats) in experiments.items():
    model = build_model(nums, cats)
    model.fit(train[nums + cats], train["goal"])

    prob = model.predict_proba(test[nums + cats])[:, 1]

    print(f"\n{name}")
    print("ROC-AUC:", round(roc_auc_score(test["goal"], prob), 4))
    print("Log loss:", round(log_loss(test["goal"], prob), 4))
    print("Brier score:", round(brier_score_loss(test["goal"], prob), 4))

    if name == "Full feature model":
        import joblib
        joblib.dump(model, "xg_model.joblib")
        print("Saved improved xg_model.joblib")
