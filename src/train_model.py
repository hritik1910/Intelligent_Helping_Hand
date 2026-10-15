
from pathlib import Path

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("data/raw/production_runs.csv")
df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])

df = df.sort_values("run_timestamp").reset_index(drop=True)

features = [
    "line_id",
    "product_id",
    "shift",
    "stencil_pressure_kpa",
    "paste_temp_c",
    "spi_height_um",
    "paste_volume_pct",
    "ambient_temp_c",
    "machine_temp_c"
]

target = "high_quality_risk"

# Use older records for training and newer records for testing.
split_at = int(len(df) * 0.80)

train = df.iloc[:split_at]
test = df.iloc[split_at:]

X_train = train[features]
X_test = test[features]

y_train = train[target].astype(int)
y_test = test[target].astype(int)

categorical = ["line_id", "product_id", "shift"]
numeric = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer([
    (
        "categories",
        OneHotEncoder(handle_unknown="ignore"),
        categorical
    ),
    ("numbers", "passthrough", numeric)
])

model = Pipeline([
    ("preprocessing", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        )
    )
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

if y_test.nunique() == 2:
    print("ROC-AUC:", roc_auc_score(y_test, probabilities))

Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/quality_risk_model.joblib"
)

print("Saved models/quality_risk_model.joblib")
