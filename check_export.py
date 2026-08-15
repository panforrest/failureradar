"""Verify the exported drop-list is syntactically valid and semantically correct."""
import ast
import json
import pathlib

import pandas as pd

src = pathlib.Path.home() / "Downloads" / "failureradar_filter.py"
text = src.read_text()

# 1. does it parse as Python?
tree = ast.parse(text)
print("syntax           OK (parses as Python)")

# 2. pull the literal set back out without executing the file
drop = None
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "FAILURERADAR_DROP":
        drop = ast.literal_eval(node.value)
print(f"hashes           {len(drop)} unique, {len(drop) == len(set(drop))} no dupes")

import re
header = int(re.search(r"# (\d+) episodes flagged", text).group(1))
print(f"header claims    {header}  -> {'match' if header == len(drop) else 'MISMATCH'}")

# 3. every hash must exist in the episode table
df = pd.read_parquet("episodes.parquet")
known = set(df["episode_hash"])
missing = drop - known
print(f"exist in table   {len(drop) - len(missing)}/{len(drop)}"
      + (f"  MISSING: {list(missing)[:3]}" if missing else ""))

# 4. every hash must be one FailureRadar actually flagged
D = json.load(open("docs/results.json"))
flagged = {e["episode_hash"] for e in D["episodes"]}
print(f"flagged in report {len(drop & flagged)}/{len(drop)}")

# 5. what does the filter actually remove?
sub = df[df["episode_hash"].isin(drop)]
print(f"\ndrop-list covers {len(sub)} rows, "
      f"{sub['num_frames'].sum() / 30 / 3600:.2f} h of runtime")
print("labs:", sub["lab"].value_counts().to_dict())
print("top tasks:", sub["task"].value_counts().head(5).to_dict())

# 6. the other two lambdas, applied to the whole corpus
kept = df[(~df["is_deleted"])
          & (df["num_frames"].fillna(0) >= 150)
          & (~df["episode_hash"].isin(drop))]
print(f"\nfull filter keeps {len(kept):,} of {len(df):,} episodes "
      f"({100 * len(kept) / len(df):.1f}%)")
print(f"  removed by is_deleted : {int(df['is_deleted'].sum()):,}")
print(f"  removed by <150 frames: {int((df['num_frames'].fillna(0) < 150).sum()):,}")
print(f"  removed by drop-list  : {len(sub):,}")

# 7. lambdas are strings in EgoVerse's DatasetFilter - confirm each compiles
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str) \
            and node.value.startswith("lambda"):
        compile(node.value, "<lambda>", "eval")
        print(f"lambda OK        {node.value}")
