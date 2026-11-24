"""Permutation importance for the tuned Random Forest (cuML's forest doesn't expose
feature_importances_). Shuffles one raw input column at a time on held-out rows and
measures how much the RMSE (log value) grows.

Usage: python -m cuml.accel src/importance.py  ->  results/rf_importance.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from train import CATEGORICAL, DATA, LOGGED, NUMERIC, OUT, SEED, TARGET, preprocessor

best = json.loads((OUT / "rf.json").read_text())["best_params"]
feats = LOGGED + NUMERIC + CATEGORICAL
train = pd.read_parquet(DATA / "train.parquet")
test = pd.read_parquet(DATA / "test.parquet").sample(15_000, random_state=SEED)
pipe = Pipeline([("prep", preprocessor(scale=False)),
                 ("model", RandomForestRegressor(random_state=SEED, n_jobs=-1, **best))])
pipe.fit(train[feats], train[TARGET])
r = permutation_importance(pipe, test[feats], test[TARGET], scoring="neg_root_mean_squared_error",
                           n_repeats=3, random_state=SEED)
imp = dict(sorted(zip(feats, map(float, r.importances_mean)), key=lambda kv: -kv[1]))
(OUT / "rf_importance.json").write_text(json.dumps(imp, indent=2))
print(json.dumps(imp, indent=2))
