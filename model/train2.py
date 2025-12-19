import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

df = pd.read_csv("data/processed/hotel_bookings_clean2.csv")

X = df[
    [
        "lead_time",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "no_of_special_requests",
        "avg_price_per_room",
        "arrival_month_num",
        "arrival_day_of_week",
        "no_of_adults",
        "no_of_children",
        "market_segment_type",
    ]
]

y = df["booking_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

num_features = [
    "lead_time",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "no_of_special_requests",
    "avg_price_per_room",
    "arrival_month_num",
    "arrival_day_of_week",
    "no_of_adults",
    "no_of_children",
]

cat_features = ["market_segment_type"]

preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
    ]
)

mlflow.set_experiment("hotel_booking_cancellation")

def run_experiment(class_weight, threshold, run_name):
    with mlflow.start_run(run_name=run_name):

        model = Pipeline(
            [
                ("prep", preprocessor),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        class_weight=class_weight,
                        random_state=42,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("threshold", threshold)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(model, "model")

        return model, accuracy, precision, recall, f1, roc_auc

model_1, acc1, prec1, rec1, f1_1, roc1 = run_experiment(
    class_weight=None,
    threshold=0.5,
    run_name="baseline"
)

model_2, acc2, prec2, rec2, f1_2, roc2 = run_experiment(
    class_weight={0: 1, 1: 2},
    threshold=0.5,
    run_name="high_recall"
)

final_model, acc, prec, rec, f1, roc = run_experiment(
    class_weight={0: 1, 1: 2},
    threshold=0.47,
    run_name="final_model"
)

os.makedirs("model", exist_ok=True)
joblib.dump(final_model, "model/hotel_model_streamlit2.pkl")

y_proba = final_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.47).astype(int)

cm = confusion_matrix(y_test, y_pred)

with open("model/metrics2.json", "w") as f:
    json.dump(
        {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
            "threshold": 0.47,
        },
        f,
        indent=4,
    )

with open("model/confusion_matrix2.json", "w") as f:
    json.dump(
        {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
        f,
        indent=4,
    )