"""Feature groups, model pipelines and evaluation helpers for glioma grade prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
SEQUENCES = ("flair", "t1", "t1ce", "t2")


# ---------- data ----------

def load_features(path) -> pd.DataFrame:
    """Load the radiomics table and add simple volume features.

    Enhancing-tumour (ET) shape features are undefined when a tumour has no enhancing part. Those cases get
    volume 0, an explicit `ET_present` flag, and 0 for the other ET shape features.
    """
    data = pd.read_csv(path)
    et_shape = [c for c in data.columns if c.startswith("ET_shape_")]
    data[et_shape] = data[et_shape].fillna(0.0)

    wt, tc, et = data["WT_voxels"], data["TC_voxels"], data["ET_voxels"]
    extra = pd.DataFrame({
        "y": (data["grade"] == "LGG").astype(int),  # positive class: low grade (the minority)
        "ET_present": (et >= 30).astype(int),
        "vol_ET_fraction_of_TC": et / tc,
        "vol_ET_fraction_of_WT": et / wt,
        "vol_TC_fraction_of_WT": tc / wt,
        "vol_necrosis_fraction_of_TC": (tc - et) / tc,
        "vol_log_WT": np.log10(wt),
    })
    return pd.concat([data, extra], axis=1).copy()  # copy: one contiguous block, no fragmentation warnings


def feature_groups(data: pd.DataFrame) -> dict[str, list[str]]:
    shape = [c for c in data.columns if "_shape_" in c]
    volume = ["vol_log_WT", "vol_ET_fraction_of_TC", "vol_ET_fraction_of_WT",
              "vol_TC_fraction_of_WT", "vol_necrosis_fraction_of_TC", "ET_present"]
    intensity_texture = [c for c in data.columns if c.split("_")[0] in SEQUENCES]
    return {
        "volumes (6)": volume,
        "shape": volume + shape,
        "intensity + texture": intensity_texture,
        "all radiomics": volume + shape + intensity_texture,
    }


def data_source(mapping: pd.DataFrame) -> pd.Series:
    """Contributing source of each case, taken from its BraTS 2019 ID (e.g. 'BraTS19_TCIA10_...')."""
    ids = mapping.set_index("BraTS_2020_subject_ID")["BraTS_2019_subject_ID"].astype(str)
    source = ids.str.extract(r"(CBICA|TCIA\d{2}|2013|TMC)")[0].fillna("new in 2020")
    source = source.where(~source.str.startswith("TCIA"), "TCIA")
    return source.rename("source")


# ---------- models ----------

def make_models() -> dict[str, object]:
    """Every model is a full pipeline, so imputation, scaling and selection are fit inside each CV fold."""
    lasso = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(l1_ratio=1.0, solver="liblinear", class_weight="balanced", max_iter=5000)),
    ])
    lasso = GridSearchCV(lasso, {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0]}, scoring="roc_auc",
                         cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE))
    forest = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample",
                                         n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    boosting = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                 class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    return {"LASSO logistic regression": lasso, "random forest": forest, "gradient boosting": boosting}


def baseline() -> DummyClassifier:
    return DummyClassifier(strategy="prior")


# ---------- evaluation ----------

def scores(y_true, probability, threshold: float = 0.5) -> dict[str, float]:
    return {
        "ROC AUC": roc_auc_score(y_true, probability),
        "balanced accuracy": balanced_accuracy_score(y_true, probability >= threshold),
        "average precision (LGG)": average_precision_score(y_true, probability),
    }


def cross_validate(model, X, y, n_splits=5, n_repeats=3) -> pd.DataFrame:
    """Repeated stratified CV; returns one row of scores per fold."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    rows = []
    for train, test in cv.split(X, y):
        fitted = clone(model).fit(X.iloc[train], y.iloc[train])
        rows.append(scores(y.iloc[test], fitted.predict_proba(X.iloc[test])[:, 1]))
    return pd.DataFrame(rows)


def bootstrap_auc(y_true, probability, n_boot=2000, seed=RANDOM_STATE) -> tuple[float, float, float]:
    """ROC AUC with a 95% percentile bootstrap interval (for small test sets)."""
    rng = np.random.default_rng(seed)
    y_true, probability = np.asarray(y_true), np.asarray(probability)
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, len(y_true), len(y_true))
        if y_true[index].min() != y_true[index].max():
            values.append(roc_auc_score(y_true[index], probability[index]))
    return roc_auc_score(y_true, probability), *np.percentile(values, [2.5, 97.5])
