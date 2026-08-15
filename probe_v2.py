"""Refined failure signals: cycle-time outliers, idle time, struggle language."""
import re
import statistics as st
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

df = pd.read_parquet("episodes.parquet")
df = df[~df["is_deleted"]]
has_seg = df["segments"].map(lambda v: isinstance(v, np.ndarray) and len(v) > 0)
seg_df = df[has_seg].copy()


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s)


def f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


IDLE = {"none", "no action", "", "idle", "nothing", "no motion"}

# Struggle language: exclude intentional placement ("drop X into/in/on <receptacle>")
STRUGGLE = re.compile(
    r"\b(again|retry|re attempt|reattempt|second attempt|missed|fail|"
    r"slip|slipped|fumble|knock over|knocked over|spill|spilled|"
    r"readjust|re adjust|redo|re do|correct|fix|accidental|"
    r"drop (on|onto) (the )?(floor|ground)|pick (it |them )?back up)\b"
)


def analyze(segs):
    items = []
    for s in segs:
        lab = norm(s.get("label", ""))
        a, b = f(s.get("start_seconds")), f(s.get("end_seconds"))
        items.append((lab, a, b, max(0.0, b - a)))

    n = len(items)
    span = max((b for _, _, b, _ in items), default=0.0)
    total_dur = sum(d for _, _, _, d in items)

    idle_time = sum(d for lab, _, _, d in items if lab in IDLE)
    idle_frac = idle_time / span if span > 0 else 0.0

    counts = Counter(lab for lab, _, _, _ in items)
    cyclicity = (max(counts.values()) / n) if n else 0.0

    # cycle-time outliers: labels repeated >=3x, segment >3x that label's median
    by_label = defaultdict(list)
    for lab, _, _, d in items:
        by_label[lab].append(d)
    n_outlier = 0
    worst_ratio = 0.0
    outlier_time = 0.0
    for lab, ds in by_label.items():
        if len(ds) >= 3 and lab not in IDLE:
            med = st.median(ds)
            if med <= 0:
                continue
            for d in ds:
                r = d / med
                if r >= 3.0 and d >= 3.0:
                    n_outlier += 1
                    outlier_time += d - med
                    worst_ratio = max(worst_ratio, r)

    struggle = sum(1 for lab, _, _, _ in items if STRUGGLE.search(lab))

    return dict(
        n_seg=n,
        span_sec=span,
        coverage=total_dur / span if span > 0 else 0.0,
        idle_time=idle_time,
        idle_frac=idle_frac,
        cyclicity=cyclicity,
        n_outlier=n_outlier,
        outlier_time=outlier_time,
        worst_ratio=worst_ratio,
        struggle_segs=struggle,
    )


feat = pd.DataFrame([analyze(s) for s in seg_df["segments"]], index=seg_df.index)
seg_df = pd.concat([seg_df, feat], axis=1)

print("=" * 70)
print("EPISODES ANALYZED:", len(seg_df))
print("=" * 70)

print("\n--- signal distributions ---")
print(seg_df[["n_seg", "span_sec", "coverage", "idle_frac", "cyclicity",
              "n_outlier", "worst_ratio", "struggle_segs"]].describe(
    percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

print("\n--- PREVALENCE ---")
tot = len(seg_df)
for name, mask in [
    ("has idle/none segment", seg_df["idle_time"] > 0),
    ("idle_frac > 10%", seg_df["idle_frac"] > 0.10),
    ("idle_frac > 25%", seg_df["idle_frac"] > 0.25),
    ("has cycle-time outlier", seg_df["n_outlier"] >= 1),
    ("has 2+ outliers", seg_df["n_outlier"] >= 2),
    ("worst_ratio >= 5x", seg_df["worst_ratio"] >= 5),
    ("struggle language", seg_df["struggle_segs"] >= 1),
    ("annotation coverage < 80%", seg_df["coverage"] < 0.80),
    ("ANY of the above", (seg_df["idle_frac"] > 0.10) | (seg_df["n_outlier"] >= 1)
     | (seg_df["struggle_segs"] >= 1) | (seg_df["coverage"] < 0.80)),
]:
    print(f"{name:28s} {int(mask.sum()):>6d}  ({100 * mask.mean():5.2f}%)")

print("\n--- idle labels seen ---")
c = Counter()
for segs in seg_df["segments"]:
    for s in segs:
        lab = norm(s.get("label", ""))
        if lab in IDLE:
            c[lab] += 1
print(c.most_common())

print("\n--- struggle language matches (top 20) ---")
c2 = Counter()
for segs in seg_df["segments"]:
    for s in segs:
        lab = norm(s.get("label", ""))
        if STRUGGLE.search(lab):
            c2[lab] += 1
for lab, n in c2.most_common(20):
    print(f"{n:>5d}  {lab}")

print("\n--- worst cycle-time outlier episodes (with mp4) ---")
cand = seg_df[(seg_df["zarr_mp4_path"] != "") & (seg_df["n_seg"] >= 6)]
for _, r in cand.nlargest(5, "worst_ratio").iterrows():
    print(f"\n{r['episode_hash']}  task={r['task']}  worst={r['worst_ratio']:.1f}x  "
          f"outliers={r['n_outlier']}  idle={r['idle_frac']:.2f}")
    labs = defaultdict(list)
    for s in r["segments"]:
        labs[norm(s.get("label", ""))].append(f(s.get("end_seconds")) - f(s.get("start_seconds")))
    for s in r["segments"]:
        lab = norm(s.get("label", ""))
        d = f(s.get("end_seconds")) - f(s.get("start_seconds"))
        med = st.median(labs[lab]) if len(labs[lab]) >= 3 else None
        flag = ""
        if med and med > 0 and d / med >= 3.0 and d >= 3.0:
            flag = f"   <== {d / med:.1f}x median"
        print(f"    {f(s.get('start_seconds')):7.1f}-{f(s.get('end_seconds')):7.1f} "
              f"({d:5.1f}s)  {str(s.get('label'))[:60]}{flag}")

print("\n--- highest idle episodes (with mp4) ---")
for _, r in cand.nlargest(4, "idle_frac").iterrows():
    print(f"\n{r['episode_hash']}  task={r['task']}  idle_frac={r['idle_frac']:.2f}  "
          f"idle={r['idle_time']:.1f}s of {r['span_sec']:.1f}s")
    for s in r["segments"][:12]:
        print(f"    {f(s.get('start_seconds')):7.1f}-{f(s.get('end_seconds')):7.1f}  "
              f"{str(s.get('label'))[:60]}")

seg_df.drop(columns=["segments"]).to_parquet("features_v2.parquet")
print("\nwrote features_v2.parquet")
