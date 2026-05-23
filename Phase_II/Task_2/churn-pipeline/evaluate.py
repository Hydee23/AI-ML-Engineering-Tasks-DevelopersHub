import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)

# ── 1. Load saved results ─────────────────────────────────────────────────────
results = joblib.load("model/results.joblib")
lr_grid     = results["lr_grid"]
rf_grid     = results["rf_grid"]
best_model  = results["best_model"]
best_name   = results["best_name"]
X_test      = results["X_test"]
y_test      = results["y_test"]

print(f"Best model: {best_name}\n")

# ── 2. Predictions ────────────────────────────────────────────────────────────
lr_model  = lr_grid.best_estimator_
rf_model  = rf_grid.best_estimator_

lr_pred   = lr_model.predict(X_test)
rf_pred   = rf_model.predict(X_test)

lr_proba  = lr_model.predict_proba(X_test)[:, 1]
rf_proba  = rf_model.predict_proba(X_test)[:, 1]

# ── 3. Print comparison ───────────────────────────────────────────────────────
print("=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(classification_report(y_test, lr_pred))

print("=" * 50)
print("RANDOM FOREST")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(classification_report(y_test, rf_pred))

# ── 4. Plot setup ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Churn Prediction — Model Evaluation", fontsize=16, fontweight="bold")
sns.set_style("whitegrid")

# ── 5. Confusion Matrix — LR ──────────────────────────────────────────────────
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
axes[0, 0].set_title("Confusion Matrix — Logistic Regression")
axes[0, 0].set_ylabel("Actual")
axes[0, 0].set_xlabel("Predicted")

# ── 6. Confusion Matrix — RF ──────────────────────────────────────────────────
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", ax=axes[0, 1],
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
axes[0, 1].set_title("Confusion Matrix — Random Forest")
axes[0, 1].set_ylabel("Actual")
axes[0, 1].set_xlabel("Predicted")

# ── 7. ROC Curve ──────────────────────────────────────────────────────────────
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba)
lr_auc = auc(lr_fpr, lr_tpr)
rf_auc = auc(rf_fpr, rf_tpr)

axes[0, 2].plot(lr_fpr, lr_tpr, label=f"LR (AUC = {lr_auc:.3f})", color="steelblue")
axes[0, 2].plot(rf_fpr, rf_tpr, label=f"RF (AUC = {rf_auc:.3f})", color="seagreen")
axes[0, 2].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[0, 2].set_title("ROC Curve")
axes[0, 2].set_xlabel("False Positive Rate")
axes[0, 2].set_ylabel("True Positive Rate")
axes[0, 2].legend()

# ── 8. Precision-Recall Curve ─────────────────────────────────────────────────
lr_prec, lr_rec, _ = precision_recall_curve(y_test, lr_proba)
rf_prec, rf_rec, _ = precision_recall_curve(y_test, rf_proba)

axes[1, 0].plot(lr_rec, lr_prec, label="Logistic Regression", color="steelblue")
axes[1, 0].plot(rf_rec, rf_prec, label="Random Forest", color="seagreen")
axes[1, 0].set_title("Precision-Recall Curve")
axes[1, 0].set_xlabel("Recall")
axes[1, 0].set_ylabel("Precision")
axes[1, 0].legend()

# ── 9. Model Comparison Bar Chart ────────────────────────────────────────────
metrics = ["Accuracy", "CV Score"]
lr_scores = [accuracy_score(y_test, lr_pred), lr_grid.best_score_]
rf_scores = [accuracy_score(y_test, rf_pred), rf_grid.best_score_]

x = np.arange(len(metrics))
width = 0.35
axes[1, 1].bar(x - width/2, lr_scores, width, label="Logistic Regression", color="steelblue")
axes[1, 1].bar(x + width/2, rf_scores, width, label="Random Forest", color="seagreen")
axes[1, 1].set_title("Model Comparison")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics)
axes[1, 1].set_ylim(0.7, 0.9)
axes[1, 1].set_ylabel("Score")
axes[1, 1].legend()
for i, (lr, rf) in enumerate(zip(lr_scores, rf_scores)):
    axes[1, 1].text(i - width/2, lr + 0.002, f"{lr:.3f}", ha="center", fontsize=9)
    axes[1, 1].text(i + width/2, rf + 0.002, f"{rf:.3f}", ha="center", fontsize=9)

# ── 10. Feature Importance (RF only) ─────────────────────────────────────────
rf_classifier = rf_model.named_steps["model"]
preprocessor  = rf_model.named_steps["preprocessor"]
feature_names = (
    preprocessor.transformers_[0][2] +
    list(preprocessor.transformers_[1][1]
         .named_steps["encoder"]
         .get_feature_names_out(preprocessor.transformers_[1][2]))
)
importances = rf_classifier.feature_importances_
top_idx = np.argsort(importances)[-15:]

axes[1, 2].barh(
    [feature_names[i] for i in top_idx],
    importances[top_idx],
    color="seagreen"
)
axes[1, 2].set_title("Top 15 Feature Importances (Random Forest)")
axes[1, 2].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("model/evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nEvaluation chart saved to model/evaluation.png")