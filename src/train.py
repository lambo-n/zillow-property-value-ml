"""Tune each model with 5-fold CV on the training set, then evaluate once on the test set.

Usage (GPU models run under cuML's scikit-learn accelerator; the MLP runs on CPU):
    python -m cuml.accel src/train.py baseline knn rf
    python src/train.py mlp

Writes results/<model>.json (best params, CV and train/test metrics),
results/cv_<model>.csv (every grid combination) and results/pred_<model>.npz
(test-set predictions, for plots).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results"
SEED = 42

# Skewed areas get log1p (the slides' outlier fix); other numerics pass through.
LOGGED = ["calculatedfinishedsquarefeet", "lotsizesquarefeet"]
NUMERIC = ["bathroomcnt", "bedroomcnt", "latitude", "longitude", "age", "has_pool"]
CATEGORICAL = ["fips", "propertylandusetypeid"]   # codes, not quantities -> one-hot
TARGET = "log_value"


def preprocessor(scale: bool) -> ColumnTransformer:
    """Median-impute, log the areas, one-hot the codes; z-score only when the model
    is distance/gradient based (k-NN, MLP, linear). Fit inside each CV fold."""
    def numeric(extra):
        steps = [("impute", SimpleImputer(strategy="median"))] + extra
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)
    return ColumnTransformer([
        ("logged", numeric([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))]), LOGGED),
        ("numeric", numeric([]), NUMERIC),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


MODELS = {
    "baseline": (True, LinearRegression(), {}),
    "knn": (True, KNeighborsRegressor(), {
        "model__n_neighbors": [5, 10, 20, 40, 80],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }),
    "rf": (False, RandomForestRegressor(random_state=SEED, n_jobs=-1), {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [8, 12, 16, 20, 24],
    }),
    "mlp": (True, MLPRegressor(activation="relu", early_stopping=True, max_iter=200,
                               random_state=SEED), {
        "model__hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32)],
        "model__learning_rate_init": [0.0003, 0.001, 0.003, 0.01],
    }),
}


def metrics(y_log: np.ndarray, p_log: np.ndarray) -> dict:
    y, p = np.exp(y_log), np.exp(p_log)
    res = p_log - y_log
    return {
        "rmse_dollars": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae_dollars": float(np.mean(np.abs(p - y))),
        "median_pct_error": float(np.median(np.abs(p - y) / y) * 100),
        "rmse_log": float(np.sqrt(np.mean(res ** 2))),
        "r2_log": float(1 - np.sum(res ** 2) / np.sum((y_log - y_log.mean()) ** 2)),
    }


def run(name: str) -> None:
    scale, model, grid = MODELS[name]
    train = pd.read_parquet(DATA / "train.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    feats = LOGGED + NUMERIC + CATEGORICAL
    Xtr, ytr = train[feats], train[TARGET].to_numpy()
    Xte, yte = test[feats], test[TARGET].to_numpy()

    pipe = Pipeline([("prep", preprocessor(scale)), ("model", model)])
    search = GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=SEED),
                          scoring="neg_root_mean_squared_error", return_train_score=True,
                          refit=True, n_jobs=8 if name == "mlp" else 1)
    t0 = time.time()
    search.fit(Xtr, ytr)
    elapsed = time.time() - t0

    cv = pd.DataFrame(search.cv_results_)
    keep = [c for c in cv.columns if c.startswith("param_")] + [
        "mean_test_score", "std_test_score", "mean_train_score", "mean_fit_time", "rank_test_score"]
    cv = cv[keep].copy()
    cv["cv_rmse_log"] = -cv.pop("mean_test_score")
    cv["train_rmse_log"] = -cv.pop("mean_train_score")
    cv.to_csv(OUT / f"cv_{name}.csv", index=False)

    best = search.best_estimator_
    p_tr, p_te = np.asarray(best.predict(Xtr)), np.asarray(best.predict(Xte))
    out = {
        "model": name,
        "best_params": {k.replace("model__", ""): (list(v) if isinstance(v, tuple) else v)
                        for k, v in search.best_params_.items()},
        "best_cv_rmse_log": float(-search.best_score_),
        "n_combinations": len(cv), "n_folds": 5, "n_fits": len(cv) * 5,
        "search_seconds": elapsed,
        "n_train": len(Xtr), "n_test": len(Xte),
        "train": metrics(ytr, p_tr), "test": metrics(yte, p_te),
        "feature_names": list(best.named_steps["prep"].get_feature_names_out()),
    }
    m = best.named_steps["model"]
    if hasattr(m, "feature_importances_"):
        try:
            fi = [float(v) for v in m.feature_importances_]
            if all(np.isfinite(fi)):  # cuML's forest can return NaN here
                out["feature_importances"] = fi
        except Exception as e:  # cuML may not expose it
            out["feature_importances_error"] = str(e)
    if hasattr(m, "loss_curve_"):
        out["loss_curve"] = [float(v) for v in m.loss_curve_]
        out["validation_scores"] = [float(v) for v in (m.validation_scores_ or [])]
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(OUT / f"pred_{name}.npz", y=yte, pred=p_te)
    print(f"[{name}] {elapsed:.0f}s  best {out['best_params']}  CV log-RMSE {out['best_cv_rmse_log']:.4f}  "
          f"test RMSE ${out['test']['rmse_dollars']:,.0f}  R2(log) {out['test']['r2_log']:.3f}", flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name in sys.argv[1:] or ["baseline", "knn", "rf"]:
        run(name)
