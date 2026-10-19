"""
train_model.py
--------------
Full training pipeline for the Lifestyle Habit Dataset.

Targets
  - Productivity_Score   -> RandomForestRegressor   (MAE, R2)
  - Lifestyle_Category   -> RandomForestClassifier  (Accuracy, Precision, Confusion Matrix)

Saved artefacts  (models/)
  - preprocessor.pkl    fitted ColumnTransformer  (impute + scale + OHE)
  - features.pkl        ordered list of raw feature column names
  - model_reg.pkl       fitted RandomForestRegressor
  - model_clf.pkl       fitted RandomForestClassifier

Single-row prediction in app.py:
    preprocessor, features, model_reg, model_clf = load_artefacts()
    row_df       = pd.DataFrame([user_input_dict])[features]
    row_proc     = preprocessor.transform(row_df)
    productivity = model_reg.predict(row_proc)[0]
    category     = model_clf.predict(row_proc)[0]
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    precision_score,
    r2_score,
)
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ==============================================================================
# Configuration
# ==============================================================================
DATA_PATH    = "data/Lifestyle_Habit_Dataset.csv"
MODELS_DIR   = "models"
TARGET_REG   = "Productivity_Score"    # continuous  -> RandomForestRegressor
TARGET_CLF   = "Lifestyle_Category"   # 3-class     -> RandomForestClassifier
RANDOM_SEED  = 42
TEST_SIZE    = 0.20
N_ESTIMATORS = 200


def section(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ==============================================================================
# 1. Load data
# ==============================================================================
def load_data(path: str) -> pd.DataFrame:
    """Load CSV from disk."""
    df = pd.read_csv(path)
    print(f"Loaded  ->  {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    return df


# ==============================================================================
# 2. Feature definition
# ==============================================================================
def define_features(df: pd.DataFrame) -> list:
    """
    Return the ordered list of feature columns the models will use.
    Both target columns are excluded; no ID / free-text columns exist
    in this dataset, so everything else is kept.
    """
    drop_cols = [TARGET_REG, TARGET_CLF]
    features = [c for c in df.columns if c not in drop_cols]
    return features


# ==============================================================================
# 3. Preprocessing pipeline
# ==============================================================================
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    ColumnTransformer with two sub-pipelines:
      numeric     -> median imputation -> StandardScaler
      categorical -> mode  imputation -> OneHotEncoder
    """
    numeric_cols     = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe,     numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# ==============================================================================
# 4. Top-N feature importance helper
# ==============================================================================
def top_n_features(model, preprocessor: ColumnTransformer, n: int = 3) -> list:
    """
    Map Random-Forest feature importances back to human-readable names.
    Returns list of (feature_name, importance) tuples, sorted descending.
    """
    feat_names   = preprocessor.get_feature_names_out()
    importances  = model.feature_importances_
    ranked = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    return ranked[:n]


# ==============================================================================
# 5. Save artefacts
# ==============================================================================
def save_artefacts(preprocessor, features, model_reg, model_clf, out_dir: str):
    """Persist all pipeline artefacts so app.py can reload them."""
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(out_dir, "preprocessor.pkl"))
    joblib.dump(features,     os.path.join(out_dir, "features.pkl"))
    joblib.dump(model_reg,    os.path.join(out_dir, "model_reg.pkl"))
    joblib.dump(model_clf,    os.path.join(out_dir, "model_clf.pkl"))

    print(f"\nArtefacts saved to  ->  {out_dir}/")
    for fname in ["preprocessor.pkl", "features.pkl", "model_reg.pkl", "model_clf.pkl"]:
        fpath = os.path.join(out_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fname:<22}  {size_kb:>8.1f} KB")


def save_confusion_matrix_png(cm, classes: list, out_dir: str):
    """
    Save a styled confusion matrix heat-map to models/confusion_matrix.png
    so app.py can load and display it without sklearn/matplotlib at runtime.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Actual",
        xlabel="Predicted",
        title="Confusion Matrix — Lifestyle_Category",
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  confusion_matrix.png   saved -> {out_path}")


# ==============================================================================
# 6. Main pipeline
# ==============================================================================
def main():

    # -- Load ------------------------------------------------------------------
    section("Step 1 | Load Data")
    df = load_data(DATA_PATH)

    # -- Feature definition ----------------------------------------------------
    section("Step 2 | Feature Columns")
    features = define_features(df)
    print(f"Total features used: {len(features)}\n")
    for i, f in enumerate(features, 1):
        dtype = str(df[f].dtype)
        print(f"  {i:>2}. {f:<28}  ({dtype})")

    X     = df[features]
    y_reg = df[TARGET_REG]   # continuous
    y_clf = df[TARGET_CLF]   # categorical

    # -- Train / test split ----------------------------------------------------
    section("Step 3 | Train / Test Split  (80/20, stratified on Lifestyle_Category)")
    X_train, X_test, y_train_clf, y_test_clf = train_test_split(
        X, y_clf,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_clf,
    )
    # Align regression targets to the same row indices
    y_train_reg = y_reg.loc[X_train.index]
    y_test_reg  = y_reg.loc[X_test.index]

    print(f"  Training samples : {len(X_train):,}")
    print(f"  Test samples     : {len(X_test):,}")
    print(f"\n  Class distribution (train):")
    for cls, cnt in y_train_clf.value_counts().items():
        print(f"    {cls:<15}  {cnt:>5,}  ({cnt / len(y_train_clf) * 100:.1f}%)")

    # -- Preprocessing ---------------------------------------------------------
    section("Step 4 | Preprocessing  (fit on train, transform both splits)")
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)
    print(f"  Processed X_train shape : {X_train_proc.shape}")
    print(f"  Processed X_test  shape : {X_test_proc.shape}")

    # -- Regressor -------------------------------------------------------------
    section(f"Step 5 | RandomForestRegressor  ->  predicting '{TARGET_REG}'")
    model_reg = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model_reg.fit(X_train_proc, y_train_reg)

    y_pred_reg = model_reg.predict(X_test_proc)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2  = r2_score(y_test_reg, y_pred_reg)

    print(f"  MAE  (Mean Absolute Error) : {mae:.4f}")
    print(f"  R2   (R-squared)           : {r2:.4f}")

    print(f"\n  Top 3 most important features:")
    for rank, (feat, imp) in enumerate(top_n_features(model_reg, preprocessor, n=3), 1):
        label = feat.split("__", 1)[-1]   # strip "num__" / "cat__" prefix
        print(f"    {rank}. {label:<35}  importance = {imp:.4f}")

    # -- Classifier ------------------------------------------------------------
    section(f"Step 6 | RandomForestClassifier  ->  predicting '{TARGET_CLF}'")
    model_clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model_clf.fit(X_train_proc, y_train_clf)

    y_pred_clf = model_clf.predict(X_test_proc)
    classes    = model_clf.classes_

    acc  = accuracy_score(y_test_clf, y_pred_clf)
    prec = precision_score(y_test_clf, y_pred_clf, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test_clf, y_pred_clf, labels=classes)

    print(f"  Accuracy            : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Precision (weighted): {prec:.4f}")

    print(f"\n  Confusion Matrix  (rows = actual, cols = predicted):")
    col_w = max(len(c) for c in classes) + 2
    header = "".join(f"{c:>{col_w}}" for c in classes)
    print(f"  {'':>{col_w}}{header}")
    for actual_cls, row in zip(classes, cm):
        row_str = "".join(f"{v:>{col_w}}" for v in row)
        print(f"  {actual_cls:>{col_w}}{row_str}")

    print(f"\n  Top 3 most important features:")
    for rank, (feat, imp) in enumerate(top_n_features(model_clf, preprocessor, n=3), 1):
        label = feat.split("__", 1)[-1]
        print(f"    {rank}. {label:<35}  importance = {imp:.4f}")

    # -- Save artefacts --------------------------------------------------------
    section("Step 7 | Save Artefacts")
    save_artefacts(preprocessor, features, model_reg, model_clf, MODELS_DIR)
    save_confusion_matrix_png(cm, list(classes), MODELS_DIR)

    # -- app.py usage guide ----------------------------------------------------
    section("app.py usage guide")
    print(
        "  from joblib import load\n"
        "  import pandas as pd\n\n"
        "  preprocessor = load('models/preprocessor.pkl')\n"
        "  features     = load('models/features.pkl')\n"
        "  model_reg    = load('models/model_reg.pkl')\n"
        "  model_clf    = load('models/model_clf.pkl')\n\n"
        "  # Build a single-row DataFrame with the exact feature columns\n"
        "  row = pd.DataFrame([user_input_dict])[features]\n\n"
        "  # Transform and predict\n"
        "  row_proc     = preprocessor.transform(row)\n"
        "  productivity = model_reg.predict(row_proc)[0]   # float\n"
        "  category     = model_clf.predict(row_proc)[0]   # 'Healthy'/'Balanced'/'Unhealthy'\n"
    )

    section("Done")


if __name__ == "__main__":
    main()
