
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

df = pd.read_csv("data/shots.csv")

# Calculate shot distance and angle to goal
dx = 120 - df["x"]
dy = 40 - df["y"]

df["distance"] = np.sqrt(dx**2 + dy**2)

lower = (36 - df["y"])
upper = (44 - df["y"])

df["angle"] = np.arctan2(
    abs(8 * dx),
    dx**2 + lower * upper
)

numeric_features = [
    "x", "y", "minute", "distance", "angle"
]

categorical_features = [
    "shot_type", "body_part", "technique", "under_pressure"
]

for col in categorical_features:
    df[col] = df[col].fillna("Unknown").astype(str)

df = df.dropna(subset=numeric_features + ["goal"])

X = df[numeric_features + categorical_features]
y = df["goal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

print("Training shots:", len(X_train))
print("Testing shots:", len(X_test))
print("Goals in dataset:", int(y.sum()))
print("Goal rate:", round(y.mean() * 100, 2), "%")
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("Log loss:", round(log_loss(y_test, probabilities), 4))
print("Brier score:", round(brier_score_loss(y_test, probabilities), 4))

joblib.dump(model, "xg_model.joblib")
print("\nModel saved as xg_model.joblib")

print("\nExample shot probabilities:")
for probability in probabilities[:5]:
    print(f"{probability:.1%}")
