"""Clean the Zillow Prize 2017 property data for assessed-value regression.

Reads  data/raw/properties_2017.csv
Writes data/processed/clean.parquet        all cleaned residential parcels
       data/processed/train.parquet        80% of a random sample (seed 42)
       data/processed/test.parquet         20% held out, never used for tuning

Usage: python src/prepare_data.py [--sample N]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "properties_2017.csv"
OUT = ROOT / "data" / "processed"
SEED = 42

TARGET = "taxvaluedollarcnt"

# Residential land-use codes (see zillow_data_dictionary.xlsx). Excludes
# commercial (31), store/office (47) and residential common area (270).
RESIDENTIAL = [246, 247, 248, 260, 261, 263, 264, 265, 266, 267, 269, 275, 279]

# Parts of, or derived from, the target -> would leak it into the features.
LEAKAGE = ["structuretaxvaluedollarcnt", "landtaxvaluedollarcnt", "taxamount",
           "assessmentyear", "taxdelinquencyflag", "taxdelinquencyyear"]

# Features recorded in all three counties.
KEEP = ["bathroomcnt", "bedroomcnt", "calculatedfinishedsquarefeet",
        "lotsizesquarefeet", "yearbuilt", "latitude", "longitude",
        "fips", "propertylandusetypeid", "regionidzip"]


def load() -> pd.DataFrame:
    return pd.read_csv(RAW, engine="pyarrow")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    log = [("raw parcels", n0)]

    df = df[df["propertylandusetypeid"].isin(RESIDENTIAL)]
    log.append(("residential land use", len(df)))

    df = df.dropna(subset=[TARGET, "calculatedfinishedsquarefeet"])
    log.append(("has target and living area", len(df)))

    # A home with 0 bathrooms is almost always a missing value coded as 0.
    df = df[df["bathroomcnt"] > 0]
    log.append(("bathroomcnt > 0", len(df)))

    # Drop implausible values: under $10k assessed or under 300 sq ft, and
    # the top 0.1% of values (extreme estates that dominate squared error).
    hi = df[TARGET].quantile(0.999)
    df = df[(df[TARGET] >= 10_000) & (df[TARGET] <= hi)
            & (df["calculatedfinishedsquarefeet"] >= 300)]
    log.append((f"value in [$10k, ${hi:,.0f}], sqft >= 300", len(df)))

    # poolcnt is 1 or blank in every county -> blank means no pool.
    out = df[KEEP].copy()
    out["has_pool"] = df["poolcnt"].notna().astype("int8")

    out["age"] = 2017 - out.pop("yearbuilt")
    out["latitude"] /= 1e6           # stored as degrees * 1e6
    out["longitude"] /= 1e6
    # Integer codes, not quantities: one-hot encode them in the model pipeline.
    for c in ["fips", "propertylandusetypeid"]:
        out[c] = out[c].astype("int64")
    out["log_value"] = np.log(df[TARGET])
    out[TARGET] = df[TARGET]
    out.index = df["parcelid"].astype("int64")

    assert not set(LEAKAGE) & set(out.columns)
    for step, n in log:
        print(f"{step:<45} {n:>10,}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300_000,
                    help="rows sampled for modeling (train + test)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = clean(load())
    df.to_parquet(OUT / "clean.parquet")

    sample = df.sample(n=min(args.sample, len(df)), random_state=SEED)
    train, test = train_test_split(sample, test_size=0.2, random_state=SEED,
                                   stratify=sample["fips"])
    train.to_parquet(OUT / "train.parquet")
    test.to_parquet(OUT / "test.parquet")

    print(f"\nclean {df.shape}, train {train.shape}, test {test.shape}")
    print("missing % in train:")
    print((train.isna().mean() * 100).round(2)[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()
