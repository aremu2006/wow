"""
src/train_model.py
==================
Full ML training pipeline:
  1. Load CSV dataset
  2. Preprocess (dedup, nulls, labels)
  3. Extract features
  4. Split 80/20
  5. Train Random Forest + SVM
  6. Evaluate (Accuracy, Confusion Matrix, Precision, Recall, F1, ROC-AUC)
  7. Hyperparameter tuning via GridSearchCV
  8. Serialize best model → models/best_model.joblib

Run from project root:
    python src/train_model.py
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.feature_extraction import extract_features, FEATURE_COLUMNS

# ─────────────────────────────────────────────────────────────────
# STEP 1 & 2 — LOAD + PREPROCESS
# ─────────────────────────────────────────────────────────────────

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    print("\n" + "═" * 60)
    print("  STEP 1-2: DATA LOADING & PREPROCESSING")
    print("═" * 60)

    df = pd.read_csv(csv_path)
    print(f"  Raw rows          : {len(df)}")

    # Drop duplicates
    df = df.drop_duplicates(subset=['url'])
    print(f"  After dedup       : {len(df)}")

    # Drop nulls
    df = df.dropna(subset=['url', 'label'])
    print(f"  After null drop   : {len(df)}")

    # Ensure binary label
    df['label'] = df['label'].astype(int).clip(0, 1)
    print(f"  Benign (0)        : {(df['label']==0).sum()}")
    print(f"  Malicious (1)     : {(df['label']==1).sum()}")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# STEP 3 & 4 — FEATURE EXTRACTION + VECTORISATION
# ─────────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame):
    print("\n  STEP 3-4: FEATURE EXTRACTION & VECTORISATION")
    print("─" * 60)

    feature_rows = []
    for url in df['url']:
        feature_rows.append(extract_features(url))

    X = pd.DataFrame(feature_rows)[FEATURE_COLUMNS].values.astype(float)
    y = df['label'].values

    # Replace any -1 (missing domain_age) with median
    col_idx = FEATURE_COLUMNS.index('domain_age_days')
    col_vals = X[:, col_idx]
    median_age = np.median(col_vals[col_vals >= 0]) if np.any(col_vals >= 0) else 365
    X[X[:, col_idx] == -1, col_idx] = median_age

    print(f"  Feature matrix    : {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Feature columns   : {FEATURE_COLUMNS}")
    return X, y


# ─────────────────────────────────────────────────────────────────
# STEP 5 — DATA SPLITTING
# ─────────────────────────────────────────────────────────────────

def split_data(X, y, test_size=0.20):
    print("\n  STEP 5: DATA SPLITTING (80/20 stratified)")
    print("─" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"  Train set         : {X_train.shape[0]} samples")
    print(f"  Test set          : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────
# STEP 6-8 — MODEL TRAINING + EVALUATION
# ─────────────────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n  STEP 6-8: MODEL TRAINING & EVALUATION")
    print("═" * 60)

    models = {
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=12,
                min_samples_leaf=2, class_weight='balanced',
                random_state=42, n_jobs=-1
            ))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel='rbf', C=1.0, gamma='scale',
                probability=True, class_weight='balanced',
                random_state=42
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, C=1.0, class_weight='balanced',
                random_state=42
            ))
        ]),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n  [{name}]")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc   = accuracy_score(y_test, y_pred)
        auc   = roc_auc_score(y_test, y_prob)
        cv_sc = cross_val_score(model, X_train, y_train,
                                cv=cv, scoring='roc_auc').mean()

        print(f"    Accuracy    : {acc:.4f}")
        print(f"    ROC-AUC     : {auc:.4f}")
        print(f"    CV-AUC (5)  : {cv_sc:.4f}")
        print(f"\n    Classification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['Benign', 'Malicious'],
                                    digits=4))

        results[name] = dict(
            model=model, y_pred=y_pred, y_prob=y_prob,
            acc=acc, auc=auc, cv_auc=cv_sc
        )

    return results


# ─────────────────────────────────────────────────────────────────
# STEP 9 — HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────

def tune_random_forest(X_train, y_train):
    print("\n  STEP 9: HYPERPARAMETER TUNING (Random Forest — GridSearchCV)")
    print("─" * 60)

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth':    [8, 12, None],
        'clf__min_samples_leaf': [1, 2],
    }

    base = Pipeline([
        ("clf", RandomForestClassifier(
            class_weight='balanced', random_state=42, n_jobs=-1
        ))
    ])

    grid = GridSearchCV(
        base, param_grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc', n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    print(f"  Best params : {grid.best_params_}")
    print(f"  Best AUC    : {grid.best_score_:.4f}")
    return grid.best_estimator_


# ─────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────

def save_plots(results, X_test, y_test, X_train, feature_cols, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)

    # --- Confusion matrices ---
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r['y_pred'])
        disp = ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malicious'])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrices.png", dpi=120)
    plt.close()
    print(f"\n  Saved → {out_dir}/confusion_matrices.png")

    # --- ROC curves ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curves.png", dpi=120)
    plt.close()
    print(f"  Saved → {out_dir}/roc_curves.png")

    # --- Feature importance (RF) ---
    if "Random Forest" in results:
        rf_pipe = results["Random Forest"]["model"]
        rf_clf  = rf_pipe.named_steps["clf"]
        importances = rf_clf.feature_importances_
        fi = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:15]
        names_fi, vals_fi = zip(*fi)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(names_fi[::-1], vals_fi[::-1], color='steelblue')
        ax.set_xlabel("Importance")
        ax.set_title("Top 15 Feature Importances (Random Forest)")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/feature_importance.png", dpi=120)
        plt.close()
        print(f"  Saved → {out_dir}/feature_importance.png")


# ─────────────────────────────────────────────────────────────────
# STEP 10 — MODEL SERIALISATION
# ─────────────────────────────────────────────────────────────────

def save_model(model, feature_cols, path="models/best_model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model, "feature_columns": feature_cols}
    joblib.dump(payload, path)
    print(f"\n  STEP 10: Model serialised → {path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    # Generate dataset if not present
    dataset_path = "data/urls_dataset.csv"
    if not os.path.exists(dataset_path):
        print("  Dataset not found — generating ...")
        os.system("python data/generate_dataset.py")

    df = load_and_preprocess(dataset_path)
    X, y = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Tune best model
    best_tuned = tune_random_forest(X_train, y_train)
    y_pred_tuned = best_tuned.predict(X_test)
    y_prob_tuned = best_tuned.predict_proba(X_test)[:, 1]
    tuned_auc = roc_auc_score(y_test, y_prob_tuned)
    tuned_acc = accuracy_score(y_test, y_pred_tuned)
    results["Tuned RF"] = dict(
        model=best_tuned, y_pred=y_pred_tuned, y_prob=y_prob_tuned,
        acc=tuned_acc, auc=tuned_auc, cv_auc=tuned_auc
    )

    print("\n  FINAL TUNED RF REPORT:")
    print(classification_report(y_test, y_pred_tuned,
                                target_names=['Benign', 'Malicious'], digits=4))

    save_plots(results, X_test, y_test, X_train, FEATURE_COLUMNS)

    # Pick best by AUC
    best_name = max(results, key=lambda n: results[n]['auc'])
    print(f"\n  Best overall model : {best_name}  (AUC={results[best_name]['auc']:.4f})")
    save_model(results[best_name]['model'], FEATURE_COLUMNS)

    print("\n  Training complete.\n")


if __name__ == "__main__":
    main()
