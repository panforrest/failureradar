import re
from collections import Counter

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 100)

df = pd.read_parquet("episodes.parquet")
df = df[~df["is_deleted"]]

has_seg = df["segments"].map(lambda v: isinstance(v, np.ndarray) and len(v) > 0)
seg_df = df[has_seg].copy()
print("episodes with segments:", len(seg_df))
print("\nsegments by lab:")
print(seg_df.groupby("lab").size().sort_values(ascending=False))
print("\nsegments by task (top 15):")
print(seg_df.groupby("task").size().sort_values(ascending=False).head(15))


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def analyze(segs):
    labels = [norm(s.get("label", "")) for s in segs]
    starts = [f(s.get("start_seconds")) for s in segs]
    ends = [f(s.get("end_seconds")) for s in segs]
    n = len(labels)
    immediate = sum(1 for i in range(1, n) if labels[i] == labels[i - 1])
    counts = Counter(labels)
    repeated_labels = sum(1 for k, v in counts.items() if v > 1)
    max_rep = max(counts.values()) if counts else 0
    # non-adjacent revisit: label reappears after at least one different label
    revisit = 0
    for i in range(1, n):
        if labels[i] != labels[i - 1] and labels[i] in labels[:i - 1]:
            revisit += 1
    dur = (ends[-1] - starts[0]) if n else 0.0
    return dict(
        n_seg=n,
        n_uniq=len(counts),
        immediate_repeats=immediate,
        revisits=revisit,
        repeated_labels=repeated_labels,
        max_label_count=max_rep,
        span_sec=dur,
        redundancy=1.0 - (len(counts) / n) if n else 0.0,
    )


rows = [analyze(s) for s in seg_df["segments"]]
feat = pd.DataFrame(rows, index=seg_df.index)
seg_df = pd.concat([seg_df, feat], axis=1)

print("\n--- retry-signal distribution over", len(seg_df), "episodes ---")
print(seg_df[["n_seg", "n_uniq", "immediate_repeats", "revisits",
              "max_label_count", "redundancy", "span_sec"]].describe())

print("\nepisodes with >=1 immediate repeat:",
      int((seg_df["immediate_repeats"] >= 1).sum()),
      f"({100 * (seg_df['immediate_repeats'] >= 1).mean():.1f}%)")
print("episodes with >=2 immediate repeats:",
      int((seg_df["immediate_repeats"] >= 2).sum()),
      f"({100 * (seg_df['immediate_repeats'] >= 2).mean():.1f}%)")
print("episodes with >=1 revisit:",
      int((seg_df["revisits"] >= 1).sum()),
      f"({100 * (seg_df['revisits'] >= 1).mean():.1f}%)")

print("\n--- immediate-repeat rate by lab ---")
print(seg_df.groupby("lab").agg(
    episodes=("n_seg", "count"),
    pct_with_repeat=("immediate_repeats", lambda s: 100 * (s >= 1).mean()),
    mean_redundancy=("redundancy", "mean"),
).sort_values("episodes", ascending=False))

# ---- explicit failure language in labels ----
FAIL = re.compile(
    r"\b(again|re ?try|retry|reattempt|second attempt|drop(s|ped|ping)?|"
    r"miss(ed|es)?|fail(ed|s|ure)?|slip(s|ped|ping)?|fumbl|knock(ed|s)? over|"
    r"spill(ed|s)?|correct(s|ed|ing)?|readjust|re ?do|fix(es|ed|ing)?|"
    r"pick(s|ed)? (it )?back up|put(s)? back)\b"
)
all_labels = Counter()
fail_hits = Counter()
ep_has_fail = []
for segs in seg_df["segments"]:
    hit = False
    for s in segs:
        lab = norm(s.get("label", ""))
        all_labels[lab] += 1
        if FAIL.search(lab):
            fail_hits[lab] += 1
            hit = True
    ep_has_fail.append(hit)
seg_df["fail_language"] = ep_has_fail

print("\n--- explicit failure language ---")
print("episodes with failure language:", int(seg_df['fail_language'].sum()),
      f"({100 * seg_df['fail_language'].mean():.2f}%)")
print("top matching labels:")
for lab, c in fail_hits.most_common(25):
    print(f"{c:>6d}  {lab}")

print("\n--- most common labels overall ---")
for lab, c in all_labels.most_common(15):
    print(f"{c:>6d}  {lab}")

print("\n--- highest immediate_repeats episodes ---")
top = seg_df.nlargest(8, "immediate_repeats")
for _, r in top.iterrows():
    print(f"\n{r['episode_hash']}  lab={r['lab']}  task={r['task']}  "
          f"n_seg={r['n_seg']} repeats={r['immediate_repeats']} mp4={bool(r['zarr_mp4_path'])}")
    for s in r["segments"][:14]:
        print(f"    {s['start_seconds']:7.1f}-{s['end_seconds']:7.1f}  {s['label']}")

seg_df.drop(columns=["segments"]).to_parquet("segments_features.parquet")
print("\nwrote segments_features.parquet")
