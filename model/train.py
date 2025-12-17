import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# =========================
# 1. Load data
# =========================
df = pd.read_csv("data/processed/hotel_bookings_clean.csv")

# =========================
# 2. Selected features (reduced, interpretable set)
# =========================
FEATURES = [
    "lead_time",
    "has_previous_cancellations",
    "booking_changes",
    "total_of_special_requests",
    "adr",
    "deposit_type",
    "customer_type"
]

X = df[FEATURES]
y = df["is_canceled"]

# =========================
# 3. Train / test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 4. Column types
# =========================
num_cols = [
    "lead_time",
    "has_previous_cancellations",
    "booking_changes",
    "total_of_special_requests",
    "adr"
]

cat_cols = [
    "deposit_type",
    "customer_type"
]

# =========================
# 5. Preprocessing
# =========================
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# =========================
# 6. Model
# =========================
model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight={0: 1, 1: 2},
        C=0.7,
        solver="liblinear",
        random_state=42
    ))
])

# =========================
# 7. Cross-validation (ROC AUC)
# =========================
cv_auc = cross_val_score(
    model,
    X_train,
    y_train,
    scoring="roc_auc",
    cv=5
)

print("CV ROC AUC:", cv_auc.mean(), "+/-", cv_auc.std())

# =========================
# 8. Train final model
# =========================
model.fit(X_train, y_train)

# =========================
# 9. Final evaluation (fixed threshold)
# =========================
FINAL_THRESHOLD = 0.58

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= FINAL_THRESHOLD).astype(int)

print("\nFinal evaluation")
print("Decision threshold:", FINAL_THRESHOLD)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# =========================
# 10. Save model
# =========================
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/hotel_model_streamlit.pkl")