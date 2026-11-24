"""Exploratory data analysis: figures for the deck and a JSON of the numbers they show.

Usage: python src/eda.py   ->  figures/*.png, results/eda.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "properties_2017.csv"
CLEAN = ROOT / "data" / "processed" / "clean.parquet"
FIG = ROOT / "figures"
OUT = ROOT / "results"
plt.style.use(Path(__file__).with_name("sage.mplstyle"))
DARK, SAGE, LIGHT, ACCENT = "#2F4A3E", "#6B9A80", "#C9DDD0", "#C98B4E"
COUNTY = {6037: "Los Angeles", 6059: "Orange", 6111: "Ventura"}
LABEL = {"calculatedfinishedsquarefeet": "living area", "bathroomcnt": "bathrooms", "bedroomcnt": "bedrooms",
         "has_pool": "has pool", "lotsizesquarefeet": "lot size", "latitude": "latitude", "longitude": "longitude",
         "age": "age", "garagecarcnt": "garage spaces", "fireplacecnt": "fireplaces",
         "buildingqualitytypeid": "building quality", "heatingorsystemtypeid": "heating type",
         "airconditioningtypeid": "air conditioning", "numberofstories": "stories", "unitcnt": "units",
         "poolcnt": "pools", "yearbuilt": "year built"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def money(v):
    return f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}k"


def target_hist(df, stats):
    v = df["taxvaluedollarcnt"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.hist(v.clip(upper=2e6), bins=80, color=SAGE)
    a.axvline(v.median(), color=DARK, ls="--", lw=1.5)
    a.axvline(v.mean(), color=ACCENT, ls="--", lw=1.5)
    a.set_title("Assessed value ($, clipped at $2M)")
    a.xaxis.set_major_formatter(lambda x, _: money(x) if x else "$0")
    a.set_ylabel("parcels")
    a.text(v.median(), a.get_ylim()[1]*0.92, f" median {money(v.median())}", color=DARK, fontsize=10)
    a.text(v.mean(), a.get_ylim()[1]*0.82, f" mean {money(v.mean())}", color=ACCENT, fontsize=10)
    b.hist(df["log_value"], bins=80, color=DARK)
    b.set_title("log(assessed value), the model target")
    b.set_xlabel("log dollars")
    for ax in (a, b):
        ax.yaxis.set_major_formatter(lambda x, _: f"{x/1e3:.0f}k" if x else "0")
    save(fig, "eda_target")
    stats["target"] = {k: float(f(v)) for k, f in
                       [("median", np.median), ("mean", np.mean), ("min", np.min), ("max", np.max)]}
    stats["target"]["skew_raw"] = float(v.skew())
    stats["target"]["skew_log"] = float(df["log_value"].skew())
    stats["target"]["share_over_1m"] = float((v > 1e6).mean())


def missing_by_county(stats):
    cols = ["garagecarcnt", "fireplacecnt", "buildingqualitytypeid", "heatingorsystemtypeid",
            "airconditioningtypeid", "numberofstories", "unitcnt", "poolcnt",
            "lotsizesquarefeet", "yearbuilt", "calculatedfinishedsquarefeet", "bathroomcnt"]
    raw = pd.read_csv(RAW, usecols=cols + ["fips"], engine="pyarrow").dropna(subset=["fips"])
    m = raw.groupby("fips")[cols].apply(lambda g: g.isna().mean() * 100).T
    m.columns = [COUNTY[int(c)] for c in m.columns]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(m.to_numpy(), cmap="Greens", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(3), m.columns)
    ax.set_yticks(range(len(cols)), [LABEL.get(c, c) for c in m.index])
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(3):
            val = m.iat[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=10,
                    color="white" if val > 55 else DARK)
    ax.set_title("Missing values by county (%)")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "eda_missing_by_county")
    stats["missing_by_county"] = m.round(1).to_dict()


def value_map(df):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    hb = ax.hexbin(df["longitude"], df["latitude"], C=df["log_value"], reduce_C_function=np.median,
                   gridsize=170, cmap="Greens", mincnt=5, linewidths=0)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    ticks = np.log([1e5, 2e5, 4e5, 8e5, 1.6e6])
    cb.set_ticks(ticks, labels=[money(np.exp(t)) for t in ticks])
    cb.set_label("median assessed value")
    ax.set_title("Median assessed value by location")
    ax.set_aspect(1.2); ax.set_axis_off()
    for name, (x, y) in {"Ventura": (-119.15, 34.52), "Los Angeles": (-118.35, 34.36),
                         "Orange": (-117.55, 33.55)}.items():
        ax.text(x, y, name, fontsize=12, color="#24332C", ha="center", fontweight="bold",
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])
    save(fig, "eda_map")


def correlations(df, stats):
    feats = ["calculatedfinishedsquarefeet", "bathroomcnt", "bedroomcnt", "has_pool",
             "lotsizesquarefeet", "latitude", "longitude", "age"]
    c = df[feats + ["log_value"]].corr()["log_value"].drop("log_value").sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.barh([LABEL[i] for i in c.index], c.values, color=[ACCENT if x < 0 else SAGE for x in c.values])
    for i, x in enumerate(c.values):
        ax.text(x + (0.01 if x >= 0 else -0.01), i, f"{x:+.2f}", va="center",
                ha="left" if x >= 0 else "right", fontsize=10)
    ax.axvline(0, color=DARK, lw=1)
    ax.set_xlim(-0.45, 0.65)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_title("Correlation with log(assessed value)")
    save(fig, "eda_correlation")
    stats["correlation_with_log_value"] = c.round(4).to_dict()


def size_and_age(df, stats):
    s = df.sample(400_000, random_state=42)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    a.hexbin(np.log10(s["calculatedfinishedsquarefeet"]), s["log_value"], gridsize=70,
             cmap="Greens", mincnt=3, bins="log", linewidths=0)
    a.set_xlabel("living area (sq ft, log scale)")
    ticks = [500, 1000, 2000, 5000, 10000]
    a.set_xticks(np.log10(ticks), [f"{t:,}" for t in ticks])
    a.set_xlim(np.log10(300), np.log10(12000))
    a.set_ylabel("log(assessed value)")
    a.set_title("Bigger homes are worth more")
    b.hexbin(s["age"], s["log_value"], gridsize=70, cmap="Greens", mincnt=3, bins="log", linewidths=0)
    med = df.groupby((df["age"] // 5) * 5)["log_value"].median()
    med = med[(med.index >= 0) & (med.index <= 110)]
    b.plot(med.index, med.values, color=ACCENT, lw=2.5, label="median by 5-year age bin")
    b.set_xlabel("age in 2017 (years)"); b.set_xlim(0, 115)
    b.set_title("Older homes are assessed lower")
    b.legend(loc="lower left")
    for ax in (a, b):
        ax.grid(False)
    save(fig, "eda_size_age")
    by_decade = df.groupby(pd.cut(df["age"], [0, 10, 20, 40, 60, 80, 200]), observed=True)["taxvaluedollarcnt"].median()
    stats["median_value_by_age"] = {str(k): float(v) for k, v in by_decade.items()}


def cv_scheme():
    fig, ax = plt.subplots(figsize=(9, 3.3))
    for r in range(5):
        for f in range(5):
            val = f == r
            ax.add_patch(plt.Rectangle((f * 1.6, 4 - r), 1.5, 0.8, color=ACCENT if val else LIGHT))
            ax.text(f * 1.6 + 0.75, 4.4 - r, "validate" if val else "train", ha="center", va="center",
                    fontsize=10, color="white" if val else DARK)
        ax.text(-0.15, 4.4 - r, f"round {r+1}", ha="right", va="center", fontsize=11)
    ax.add_patch(plt.Rectangle((8.4, 0), 1.5, 4.8, color=DARK))
    ax.text(9.15, 2.4, "test\n(20%)\nheld out", ha="center", va="center", color="white", fontsize=11)
    ax.set_xlim(-1.5, 10); ax.set_ylim(-0.2, 5)
    ax.axis("off")
    save(fig, "cv_scheme")


def main():
    FIG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    df = pd.read_parquet(CLEAN)
    stats = {"n_clean": len(df)}
    target_hist(df, stats)
    correlations(df, stats)
    value_map(df)
    size_and_age(df, stats)
    missing_by_county(stats)
    cv_scheme()
    (OUT / "eda.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
