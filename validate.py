"""Does the annotation-derived flag predict anomalous MOTION?

Compares cycle-time-outlier segments against same-label baseline segments
from the same episode, using end-effector kinematics the flag never saw.
"""
import statistics as st
from collections import defaultdict

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

seg = pd.read_parquet("kinematics_segments.parquet")
meta = pd.read_parquet("scan_meta.parquet")
print("segments:", len(seg), " episodes:", seg["episode_hash"].nunique())

# Recompute the annotation flag on exactly the segments we have kinematics for.
seg["label_n"] = seg["label"].fillna("").str.strip().str.lower()
IDLE = {"none", "no action", "", "idle", "nothing"}

med = (seg[~seg["label_n"].isin(IDLE)]
       .groupby(["episode_hash", "label_n"])["dur"]
       .agg(["median", "count"])
       .rename(columns={"median": "med_dur", "count": "n_rep"}))
seg = seg.merge(med, on=["episode_hash", "label_n"], how="left")
seg["ratio"] = seg["dur"] / seg["med_dur"]
seg["is_outlier"] = (seg["n_rep"] >= 3) & (seg["ratio"] >= 3.0) & (seg["dur"] >= 3.0)
seg["is_idle"] = seg["label_n"].isin(IDLE)
seg["is_baseline"] = (seg["n_rep"] >= 3) & (~seg["is_outlier"]) & (~seg["is_idle"])

print("\noutlier segments:", int(seg["is_outlier"].sum()))
print("baseline segments (same-label repeats):", int(seg["is_baseline"].sum()))
print("idle segments:", int(seg["is_idle"].sum()))

METRICS = ["mean_speed", "med_speed", "dwell_frac", "efficiency", "path_len"]

print("\n" + "=" * 78)
print("OUTLIER vs SAME-LABEL BASELINE  (paired within episode+label)")
print("=" * 78)

pairs = defaultdict(list)
for (ep, lab), grp in seg[seg["n_rep"] >= 3].groupby(["episode_hash", "label_n"]):
    o = grp[grp["is_outlier"]]
    b = grp[~grp["is_outlier"] & ~grp["is_idle"]]
    if len(o) == 0 or len(b) < 2:
        continue
    for m in METRICS:
        pairs[m].append((float(o[m].mean()), float(b[m].median())))

print(f"paired comparisons: {len(pairs['dwell_frac'])}")
print(f"\n{'metric':14s} {'outlier':>10s} {'baseline':>10s} {'delta':>9s} "
      f"{'win-rate':>9s}")
for m in METRICS:
    p = pairs[m]
    o = np.array([x[0] for x in p])
    b = np.array([x[1] for x in p])
    if m in ("dwell_frac", "path_len"):
        win = float((o > b).mean())
    else:
        win = float((o < b).mean())
    print(f"{m:14s} {o.mean():10.4f} {b.mean():10.4f} "
          f"{o.mean() - b.mean():+9.4f} {100 * win:8.1f}%")

# Wilcoxon-style sign test on dwell
o = np.array([x[0] for x in pairs["dwell_frac"]])
b = np.array([x[1] for x in pairs["dwell_frac"]])
n_pos = int((o > b).sum())
n_tot = int((o != b).sum())
from math import comb
p_val = sum(comb(n_tot, k) for k in range(n_pos, n_tot + 1)) / (2 ** n_tot)
print(f"\nsign test on dwell_frac: {n_pos}/{n_tot} outliers dwell more, p={p_val:.3g}")

print("\n" + "=" * 78)
print("EPISODE-LEVEL: flagged vs clean")
print("=" * 78)
ep = seg.groupby("episode_hash").agg(
    dwell=("dwell_frac", "mean"),
    speed=("mean_speed", "mean"),
    eff=("efficiency", "mean"),
    n_out=("is_outlier", "sum"),
    n_idle=("is_idle", "sum"),
).reset_index()
ep = ep.merge(meta[["episode_hash", "n_outlier", "idle_frac", "task"]],
              on="episode_hash", how="left")
ep["group"] = np.where((ep["n_outlier"] >= 1) | (ep["idle_frac"] > 0.10),
                       "flagged", "clean")
print(ep.groupby("group")[["dwell", "speed", "eff"]].agg(["mean", "median", "count"]))

print("\n--- idle ('None'-labelled) segments vs labelled segments ---")
print(seg.groupby("is_idle")[["mean_speed", "dwell_frac", "efficiency"]].mean())

print("\n--- do 'None' episodes actually contain motion? ---")
idle_eps = seg[seg["is_idle"]].groupby("episode_hash")["mean_speed"].mean()
print(idle_eps.describe())
print("null-labelled episodes whose hands ARE moving (speed > median of all):",
      int((idle_eps > seg["mean_speed"].median()).sum()), "/", len(idle_eps))

print("\n--- worst outliers with kinematics ---")
w = seg[seg["is_outlier"]].nlargest(12, "ratio")
print(w[["episode_hash", "task", "label", "dur", "med_dur", "ratio",
         "dwell_frac", "mean_speed", "efficiency"]].to_string(index=False))
