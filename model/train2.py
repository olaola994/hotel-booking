import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)


df = pd.read_csv("data/processed/hotel_bookings_clean2.csv")

NUM_FEATURES = [
    "lead_time",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "no_of_special_requests",
    "avg_price_per_room",
    "arrival_month_num",
    "arrival_day_of_week",
    "no_of_adults",
    "no_of_children"
]

CAT_FEATURES = [
    "market_segment_type"
]

FEATURES = NUM_FEATURES + CAT_FEATURES

X = df[FEATURES]
y = df["booking_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

num_pipe = Pipeline([
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        drop="first"
    ))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, NUM_FEATURES),
    ("cat", cat_pipe, CAT_FEATURES)
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight={0: 1, 1: 2},
        C=0.8,
        solver="liblinear",
        random_state=42
    ))
])

cv_auc = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc"
)

print(f"CV ROC AUC (v3): {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

model.fit(X_train, y_train)

FINAL_THRESHOLD = 0.47

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= FINAL_THRESHOLD).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\nFinal evaluation (v3)")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("ROC AUC:", roc_auc)

cm = confusion_matrix(y_test, y_pred)

confusion = {
    "true_negative": int(cm[0, 0]),
    "false_positive": int(cm[0, 1]),
    "false_negative": int(cm[1, 0]),
    "true_positive": int(cm[1, 1]),
}

ohe = model.named_steps["prep"] \
           .named_transformers_["cat"] \
           .named_steps["encoder"]

cat_feature_names = ohe.get_feature_names_out(CAT_FEATURES)

feature_names = NUM_FEATURES + list(cat_feature_names)
coefficients = model.named_steps["clf"].coef_[0]

feature_importance = dict(
    zip(feature_names, coefficients)
)

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/hotel_model_streamlit2.pkl")

with open("model/metrics2.json", "w") as f:
    json.dump({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "threshold": FINAL_THRESHOLD
    }, f, indent=4)

with open("model/confusion_matrix2.json", "w") as f:
    json.dump(confusion, f, indent=4)

with open("model/feature_importance2.json", "w") as f:
    json.dump(feature_importance, f, indent=4)
