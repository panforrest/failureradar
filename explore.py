import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)

df = pd.read_parquet("episodes.parquet")
print("ROWS:", len(df))
print("\n--- dtypes ---")
print(df.dtypes)

print("\n--- null / empty rate per column ---")
for c in df.columns:
    if c == "segments":
        continue
    n_null = df[c].isna().sum()
    try:
        n_empty = int((df[c] == "").sum()) if df[c].dtype == object else 0
    except Exception:
        n_empty = -1
    print(f"{c:26s} null={n_null:>7d}  empty={n_empty:>7d}")

print("\n--- is_deleted ---")
print(df["is_deleted"].value_counts(dropna=False))

print("\n--- is_eval / eval_success / eval_score ---")
print(df["is_eval"].value_counts(dropna=False))
print(df["eval_success"].value_counts(dropna=False))
print(df["eval_score"].describe())

print("\n--- embodiment ---")
print(df["embodiment"].value_counts(dropna=False))

print("\n--- rig_name ---")
print(df["rig_name"].value_counts(dropna=False).head(15))

print("\n--- license ---")
print(df["license"].value_counts(dropna=False).head(10))

print("\n--- num_frames describe ---")
print(df["num_frames"].describe())
print("duration_sec @30fps quantiles:")
print((df["num_frames"] / 30).quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

print("\n--- SEGMENTS ---")
seg = df["segments"]
print("python types seen:", seg.map(lambda v: type(v).__name__).value_counts().head())


def seg_len(v):
    try:
        return len(v)
    except Exception:
        return -1


lens = seg.map(seg_len)
print("segment-count distribution:")
print(lens.value_counts().head(10))
print("episodes with >=1 segment:", int((lens > 0).sum()))
print("total segments:", int(lens[lens > 0].sum()))

for v in seg[lens > 0].head(2):
    print("=== example episode segments ===")
    print(repr(v)[:2000])

print("\n--- zarr_processing_error nonempty ---")
err = df[df["zarr_processing_error"].fillna("") != ""]
print("count:", len(err))
print(err["zarr_processing_error"].value_counts().head(5))

print("\n--- unique operators ---")
print("n_operators:", df["operator"].nunique())
print(df["operator"].value_counts().head(5))

print("\n--- per-lab summary ---")
g = df.groupby("lab").agg(
    episodes=("episode_hash", "count"),
    hours=("num_frames", lambda s: s.sum() / 30 / 3600),
    med_frames=("num_frames", "median"),
    n_tasks=("task", "nunique"),
    n_ops=("operator", "nunique"),
)
print(g.sort_values("episodes", ascending=False))

print("\n--- candidate slices (lab x task) ---")
lt = df.groupby(["lab", "task"]).size().sort_values(ascending=False).head(25)
print(lt)
