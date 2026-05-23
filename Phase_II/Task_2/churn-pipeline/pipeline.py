import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ── 0. Create model folder ────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

# ── 1. Load dataset ──────────────────────────────────────────────────────────
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(URL)
print(f"Dataset loaded: {df.shape}")

# ── 2. Basic cleanup ──────────────────────────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.drop(columns=["customerID"], inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ── 3. Split features and target ──────────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Preprocessing ──────────────────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ── 5. Build pipelines ────────────────────────────────────────────────────────
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

# ── 6. GridSearchCV ───────────────────────────────────────────────────────────
print("\nTuning Logistic Regression...")
lr_params = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__solver": ["lbfgs", "liblinear"]
}
lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=5, scoring="accuracy", n_jobs=-1)
lr_grid.fit(X_train, y_train)
print(f"Best LR params: {lr_grid.best_params_}")
print(f"Best LR CV accuracy: {lr_grid.best_score_:.4f}")

print("\nTuning Random Forest...")
rf_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}
rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring="accuracy", n_jobs=-1)
rf_grid.fit(X_train, y_train)
print(f"Best RF params: {rf_grid.best_params_}")
print(f"Best RF CV accuracy: {rf_grid.best_score_:.4f}")

# ── 7. Pick best model ────────────────────────────────────────────────────────
if lr_grid.best_score_ >= rf_grid.best_score_:
    best_model = lr_grid.best_estimator_
    best_name = "Logistic Regression"
else:
    best_model = rf_grid.best_estimator_
    best_name = "Random Forest"

print(f"\nBest model: {best_name}")

# ── 8. Evaluate on test set ───────────────────────────────────────────────────
y_pred = best_model.predict(X_test)
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 9. Save results for evaluate.py ──────────────────────────────────────────
joblib.dump({
    "lr_grid": lr_grid,
    "rf_grid": rf_grid,
    "best_model": best_model,
    "best_name": best_name,
    "X_test": X_test,
    "y_test": y_test
}, "model/results.joblib")

# ── 10. Export best pipeline ──────────────────────────────────────────────────
joblib.dump(best_model, "model/churn_pipeline.joblib")
print("\nPipeline exported to model/churn_pipeline.joblib")