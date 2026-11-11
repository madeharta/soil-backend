import sys
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

W = ["humidity", "dew_point_2m", "precipitation", "rain", "weather_code",
     "evapotranspiration", "wind_speed", "wind_direction"]

for name, target in [("moisture", "moisture"), ("suction", "suction_pressure")]:
    df = pd.read_csv(f"{sys.argv[1]}/{name}.csv")
    print("=" * 30, name, df.shape)
    print(df.describe().T[["count", "mean", "std", "min", "max"]].round(3))
    df["loc"] = df["combine_loc"]
    # order of locations as stored (sessions)
    seg = (df["loc"] != df["loc"].shift()).cumsum()
    print("\nsessions in stored order (location, n rows):")
    print(df.groupby(seg).agg(loc=("loc", "first"), n=("loc", "size")).to_string())
    g = df.groupby("loc")
    print("\nper location:")
    print(g.agg(n=(target, "size"), mean=(target, "mean"), sd=(target, "std"),
                mn=(target, "min"), mx=(target, "max"),
                temp_mean=("temperature", "mean"), temp_sd=("temperature", "std"),
                n_weather_states=("humidity", lambda s: 0)).round(3).drop(columns="n_weather_states"))
    nstates = g.apply(lambda d: d[W].drop_duplicates().shape[0], include_groups=False)
    print("\ndistinct weather vectors per location:\n", nstates.to_string())
    print("distinct weather vectors overall:", df[W].drop_duplicates().shape[0])
    tot = df[target].var(ddof=0)
    within = g[target].apply(lambda s: s.var(ddof=0)).mul(g.size()).sum() / len(df)
    print(f"\nvariance total={tot:.4f}  within-location={within:.4f}  share within={within/tot:.4%}")
    print("exact duplicate rows:", df.duplicated().sum())
    print("rows with zero target:", (df[target] == 0).sum())
    print("\nPearson r with target (pooled):")
    print(df[["temperature"] + W].corrwith(df[target]).round(2).to_string())
    print("\nPearson r with target WITHIN locations (demeaned):")
    dm = df[["temperature"] + W + [target]] - g[["temperature"] + W + [target]].transform("mean")
    print(dm[["temperature"] + W].corrwith(dm[target]).round(2).to_string())
    if name == "moisture":
        both = df[target]
        print("\nsuction column in moisture file: nonzero rows", (df["suction_pressure"] != 0).sum())
