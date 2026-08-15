"""Produce web/results.json: corpus audit + failure prevalence + validation."""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from failureradar import IDLE_LABELS, score_episode

FPS = 30.0
OUT = "docs/results.json"

df = pd.read_parquet("episodes.parquet")
total_rows = len(df)

# ---------------------------------------------------------------- corpus audit
live = df[~df["is_deleted"]].copy()
live["hours"] = live["num_frames"].fillna(0) / FPS / 3600

def pct(n):
    return round(100 * n / total_rows, 3)

audit = []

def add(id_, title, n, severity, detail, fix=None):
    audit.append({"id": id_, "title": title, "count": int(n), "pct": pct(n),
                  "severity": severity, "detail": detail, "fix": fix})

add("deleted", "Episodes marked is_deleted still present in the index",
    int(df["is_deleted"].sum()), "info",
    "These rows remain in the episode table; any pipeline that forgets the "
    "is_deleted filter will silently train on retired data.",
    "df = df[~df['is_deleted']]")

bad_frames = int((df["num_frames"].fillna(-1) <= 0).sum())
add("badframes", "Episodes with non-positive num_frames", bad_frames, "high",
    "num_frames goes as low as -2, which is not a valid episode length.",
    "Filter num_frames > 0 before sampling.")

tiny = int((df["num_frames"].fillna(0).between(1, 150)).sum())
add("tiny", "Episodes shorter than 5 seconds", tiny, "medium",
    "At 30 fps these are under 150 frames - too short to contain a complete "
    "manipulation demonstration.", "Filter num_frames >= 150.")

no_mp4 = int((df["zarr_mp4_path"] == "").sum())
add("nomp4", "Episodes with no preview MP4", no_mp4, "medium",
    "No preview means no human can spot-check these episodes; they are "
    "invisible to every QA workflow including this one.")

no_zarr = int((df["zarr_processed_path"] == "").sum())
add("nozarr", "Episodes with no processed zarr path", no_zarr, "high",
    "Indexed but not downloadable.")

proc_err = int((df["zarr_processing_error"].fillna("") != "").sum())
add("procerr", "Episodes with a recorded processing error", proc_err, "info",
    "Mostly 'Zero Frames'. Present in the index regardless.")

# --- contract violations against CONTRIBUTING_DATA.md ---
no_op = int(df["operator"].isna().sum() + (df["operator"] == "").sum())
add("nooperator", "Episodes with no operator attribution", no_op, "high",
    "CONTRIBUTING_DATA.md 5.1 requires a hashed operator ID per episode. "
    "Without it you cannot detect operator-specific bias, nor exclude one "
    "demonstrator's data if it turns out to be low quality.",
    "Backfill operator IDs at ingest.")

# raw (unhashed) operator names: a sha256 hex digest is 64 hex chars
op = df["operator"].dropna()
op = op[op != ""]
raw_ops = op[~op.str.fullmatch(r"[0-9a-f]{64}", case=False, na=False)]
add("rawoperator", "Episodes whose operator field is not a hash",
    len(raw_ops), "high",
    "The guide says operator 'MUST be hashed before insertion - never store "
    "raw names/emails'. Values in the table include plain strings such as "
    f"{sorted(set(raw_ops.unique()))[:4]}.",
    "Hash these values and re-upload.")

no_lic = int(df["license"].isna().sum())
add("nolicense", "Episodes with no license field", no_lic, "high",
    "The consortium redistributes under CC BY-SA 4.0, but most episodes carry "
    "no license string at all, which makes downstream redistribution "
    "ambiguous.")

# --- task taxonomy fragmentation ---
task_counts = live.groupby("lab")["task"].nunique().to_dict()
ep_counts = live.groupby("lab").size().to_dict()


def canon(t):
    t = re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()
    words = []
    for w in t.split():
        if w.endswith("ing") and len(w) > 5:
            w = w[:-3]
        if w.endswith("es") and len(w) > 4:
            w = w[:-2]
        elif w.endswith("s") and len(w) > 3:
            w = w[:-1]
        words.append(w)
    return " ".join(sorted(words))


live["task_canon"] = live["task"].map(canon)
groups = live.groupby("task_canon")["task"].nunique()
collapsible = groups[groups > 1]
affected = int(live[live["task_canon"].isin(collapsible.index)].shape[0])
add("taskfrag", "Episodes whose task name collapses onto another task name",
    affected, "high",
    f"{live['task'].nunique():,} distinct task strings reduce to "
    f"{live['task_canon'].nunique():,} after stemming - e.g. prepare_onion vs "
    "prepare_onions, iron_clothes vs ironing_clothes. The guide asks for "
    "canonical categories, 'not a one-off trial description'. Fragmented task "
    "names break every per-task sampling or balancing strategy.",
    "Canonicalise task strings at ingest.")

singleton_tasks = live.groupby("task").size()
singletons = int((singleton_tasks == 1).sum())
add("tasksingleton", "Task names used by exactly one episode", singletons,
    "medium",
    "A task category with a single member cannot be balanced, held out, or "
    "evaluated. These are trial descriptions masquerading as categories.")

# eval columns are inert
add("evaldead", "Episodes where eval_success is the default value",
    int((df["eval_success"] == True).sum()), "info",  # noqa: E712
    "eval_success is True for 428,310 rows, False for 6, and eval_score is "
    "-1 almost everywhere, while is_eval is False for every row. These "
    "columns carry no usable supervision - there is no ground-truth "
    "success label anywhere in the dataset.")

# ------------------------------------------------- annotation-level prevalence
ann = live[live["segments"].map(lambda v: isinstance(v, np.ndarray) and len(v) > 0)]
print("annotated episodes:", len(ann))

recs = []
for _, r in ann.iterrows():
    feats, items = score_episode(r["segments"])
    recs.append({
        "episode_hash": r["episode_hash"],
        "lab": r["lab"],
        "task": r["task"],
        "num_frames": r["num_frames"],
        "mp4": r["zarr_mp4_path"],
        **{k: v for k, v in feats.items() if k != "reasons"},
        "reasons": feats["reasons"],
        "n_items": len(items),
    })
S = pd.DataFrame(recs)

ann_hours = float(ann["num_frames"].fillna(0).sum() / FPS / 3600)
lost_hours = float(S["lost_sec"].sum() / 3600)

prevalence = {
    "annotated_episodes": int(len(S)),
    "annotated_hours": round(ann_hours, 1),
    "total_segments": int(S["n_seg"].sum()),
    "flagged": int(S["flag"].sum()),
    "flagged_pct": round(100 * S["flag"].mean(), 2),
    "lost_hours": round(lost_hours, 2),
    "lost_pct_of_annotated": round(100 * lost_hours / ann_hours, 2),
    "buckets": [
        {"id": "unlabelled_all", "label": "Entirely unlabelled (every span null)",
         "n": int((S["idle_frac"] >= 0.99).sum())},
        {"id": "idle_heavy", "label": "More than 10% of runtime unlabelled",
         "n": int((S["idle_frac"] > 0.10).sum())},
        {"id": "outlier", "label": "Contains a cycle-time outlier (>=3x median)",
         "n": int((S["n_outlier"] >= 1).sum())},
        {"id": "outlier_severe", "label": "Contains a >=5x cycle-time outlier",
         "n": int((S["worst_ratio"] >= 5).sum())},
        {"id": "struggle", "label": "Explicit recovery language in a label",
         "n": int((S["n_struggle"] >= 1).sum())},
        {"id": "lowcov", "label": "Under 80% annotation coverage",
         "n": int((S["coverage"] < 0.80).sum())},
    ],
}

by_task = (S.groupby("task")
           .agg(episodes=("episode_hash", "count"),
                flagged=("flag", "sum"),
                lost_sec=("lost_sec", "sum"),
                worst=("worst_ratio", "max"))
           .reset_index())
by_task["flag_pct"] = (100 * by_task["flagged"] / by_task["episodes"]).round(1)
by_task["lost_min"] = (by_task["lost_sec"] / 60).round(1)
by_task = by_task[by_task["episodes"] >= 20].sort_values(
    "flag_pct", ascending=False)

# ------------------------------------------------------------- validation math
kin = pd.read_parquet("kinematics_segments.parquet")
kin["label_n"] = kin["label"].fillna("").str.strip().str.lower()
m = (kin[~kin["label_n"].isin(IDLE_LABELS)]
     .groupby(["episode_hash", "label_n"])["dur"]
     .agg(["median", "count"]).rename(columns={"median": "med_dur", "count": "n_rep"}))
kin = kin.merge(m, on=["episode_hash", "label_n"], how="left")
kin["ratio"] = kin["dur"] / kin["med_dur"]
kin["is_outlier"] = (kin["n_rep"] >= 3) & (kin["ratio"] >= 3.0) & (kin["dur"] >= 3.0)
kin["is_idle"] = kin["label_n"].isin(IDLE_LABELS)
work = kin[~kin["is_idle"]]

ctl = work[~work["is_outlier"]]
x = np.log(ctl["dur"].values)
y = np.log(ctl["efficiency"].values.clip(1e-4))
coef = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]


def resid(d):
    return (np.log(d["efficiency"].values.clip(1e-4))
            - (coef[0] * np.log(d["dur"].values) + coef[1]))


r_out, r_ctl = resid(work[work["is_outlier"]]), resid(ctl)

bins = [3, 5, 8, 12, 20, 35, 60, 10000]
work = work.copy()
work["bin"] = pd.cut(work["dur"], bins)
binrows = []
for b, g in work.groupby("bin", observed=True):
    o, c = g[g["is_outlier"]], g[~g["is_outlier"]]
    if len(o) >= 5 and len(c) >= 5:
        binrows.append({
            "bin": f"{int(b.left)}-{int(b.right)}s" if b.right < 1000 else "60s+",
            "n_flagged": int(len(o)), "n_normal": int(len(c)),
            "eff_flagged": round(float(o["efficiency"].median()), 4),
            "eff_normal": round(float(c["efficiency"].median()), 4),
        })

idle_k, lab_k = kin[kin["is_idle"]], kin[~kin["is_idle"]]
thr = float(lab_k["mean_speed"].quantile(0.25))
active = idle_k[idle_k["mean_speed"] > thr]

validation = {
    "scanned_episodes": int(kin["episode_hash"].nunique()),
    "scanned_segments": int(len(kin)),
    "bytes_per_episode_pct": 1.6,
    "flagged_segments": int(work["is_outlier"].sum()),
    "duration_bins": binrows,
    "trend": {"slope": round(float(coef[0]), 3), "intercept": round(float(coef[1]), 3)},
    "residual_flagged_median": round(float(np.median(r_out)), 3),
    "residual_normal_median": round(float(np.median(r_ctl)), 3),
    "efficiency_penalty": round(
        float(np.exp(np.median(r_ctl) - np.median(r_out))), 2),
    "unlabelled": {
        "segments": int(len(idle_k)),
        "hours": round(float(idle_k["dur"].sum() / 3600), 2),
        "active_segments": int(len(active)),
        "active_pct": round(100 * len(active) / max(len(idle_k), 1), 1),
        "active_hours": round(float(active["dur"].sum() / 3600), 2),
        "speed_threshold": round(thr, 5),
    },
}

# ------------------------------------------------------------------- episodes
featured = json.load(open("docs/clips/featured/manifest.json"))
featured_hashes = {f["episode_hash"] for f in featured}

seg_lookup = {r["episode_hash"]: r["segments"] for _, r in ann.iterrows()}
top = S[S["flag"]].sort_values("lost_sec", ascending=False).head(250)
episodes = []
for _, r in top.iterrows():
    feats, items = score_episode(seg_lookup[r["episode_hash"]])
    episodes.append({
        "episode_hash": r["episode_hash"],
        "task": r["task"], "lab": r["lab"],
        "duration": round(float(feats["span"]), 1),
        "idle_frac": round(float(feats["idle_frac"]), 3),
        "n_outlier": int(feats["n_outlier"]),
        "worst_ratio": round(float(feats["worst_ratio"]), 1),
        "lost_sec": round(float(feats["lost_sec"]), 1),
        "coverage": round(float(feats["coverage"]), 3),
        "reasons": feats["reasons"],
        "featured": r["episode_hash"] in featured_hashes,
        "segments": [
            {"label": i["raw"], "start": round(i["start"], 2),
             "end": round(i["end"], 2), "dur": round(i["dur"], 2),
             "ratio": round(i["ratio"], 2) if i.get("ratio") else None,
             "outlier": bool(i.get("is_outlier")),
             "idle": bool(i.get("is_idle")),
             "struggle": bool(i.get("is_struggle"))}
            for i in items
        ],
    })

for f in featured:
    feats, items = score_episode(
        [{"label": s["label"], "start_seconds": s["start"],
          "end_seconds": s["end"]} for s in f["segments"]])
    f["flags"] = {
        "idle_frac": round(float(feats["idle_frac"]), 3),
        "n_outlier": int(feats["n_outlier"]),
        "worst_ratio": round(float(feats["worst_ratio"]), 1),
        "lost_sec": round(float(feats["lost_sec"]), 1),
        "reasons": feats["reasons"],
    }
    f["segments"] = [
        {"label": i["raw"], "start": round(i["start"], 2),
         "end": round(i["end"], 2), "dur": round(i["dur"], 2),
         "ratio": round(i["ratio"], 2) if i.get("ratio") else None,
         "outlier": bool(i.get("is_outlier")), "idle": bool(i.get("is_idle"))}
        for i in items
    ]

labs = (live.groupby("lab")
        .agg(episodes=("episode_hash", "count"), hours=("hours", "sum"),
             tasks=("task", "nunique"))
        .reset_index().sort_values("episodes", ascending=False))
labs["hours"] = labs["hours"].round(1)
labs["annotated"] = labs["lab"].map(ann.groupby("lab").size()).fillna(0).astype(int)

out = {
    "corpus": {
        "episodes": int(total_rows),
        "live_episodes": int(len(live)),
        "hours": round(float(live["hours"].sum()), 1),
        "labs": labs.to_dict("records"),
        "tasks": int(live["task"].nunique()),
        "tasks_canonical": int(live["task_canon"].nunique()),
    },
    "audit": sorted(audit, key=lambda a: -a["count"]),
    "prevalence": prevalence,
    "by_task": by_task.head(40).to_dict("records"),
    "validation": validation,
    "episodes": episodes,
    "featured": featured,
}

with open(OUT, "w") as fh:
    json.dump(out, fh, separators=(",", ":"))

import os
print(f"\nwrote {OUT}  ({os.path.getsize(OUT) / 1e6:.2f} MB)")
print(json.dumps({"corpus": out["corpus"]["episodes"],
                  "hours": out["corpus"]["hours"],
                  "annotated": prevalence["annotated_episodes"],
                  "flagged": prevalence["flagged"],
                  "flagged_pct": prevalence["flagged_pct"],
                  "lost_hours": prevalence["lost_hours"],
                  "efficiency_penalty": validation["efficiency_penalty"]}, indent=2))
