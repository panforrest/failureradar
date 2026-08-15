"""Sanity-check that results.json has every field index.html reads."""
import json
import os
import urllib.request

BASE = "http://localhost:8788"

for path in ("/", "/results.json"):
    with urllib.request.urlopen(BASE + path) as r:
        print(f"{path:16s} HTTP {r.status}  {r.headers.get('Content-Length')} bytes")

D = json.load(open("docs/results.json"))

req = {
    "corpus": ["episodes", "live_episodes", "hours", "labs", "tasks", "tasks_canonical"],
    "prevalence": ["annotated_episodes", "annotated_hours", "total_segments",
                   "flagged", "flagged_pct", "lost_hours", "lost_pct_of_annotated",
                   "buckets"],
    "validation": ["scanned_episodes", "scanned_segments", "bytes_per_episode_pct",
                   "flagged_segments", "duration_bins", "trend",
                   "residual_flagged_median", "residual_normal_median",
                   "efficiency_penalty", "unlabelled"],
}
bad = []
for k, fields in req.items():
    if k not in D:
        bad.append(f"missing top-level {k}")
        continue
    for f in fields:
        if f not in D[k]:
            bad.append(f"missing {k}.{f}")

for k in ("audit", "by_task", "episodes", "featured"):
    if not D.get(k):
        bad.append(f"empty {k}")

# featured episodes: video + trace integrity
for f in D["featured"]:
    p = os.path.join("docs", "clips", "featured", f["mp4"] or "")
    if not f.get("mp4") or not os.path.exists(p):
        bad.append(f"missing mp4 for {f['episode_hash']}")
    if len(f["trace_t"]) != len(f["trace_speed"]):
        bad.append(f"trace length mismatch {f['episode_hash']}")
    if not f["segments"]:
        bad.append(f"no segments {f['episode_hash']}")
    for s in f["segments"]:
        if s["start"] is None or s["end"] is None:
            bad.append(f"null span bounds {f['episode_hash']}")
    if "flags" not in f:
        bad.append(f"no flags {f['episode_hash']}")

for e in D["episodes"]:
    for f in ("episode_hash", "task", "duration", "idle_frac", "worst_ratio",
              "lost_sec", "reasons", "segments"):
        if f not in e:
            bad.append(f"episode missing {f}")
            break

# scan the serialized text for values that render as literal junk
raw = open("docs/results.json").read()
for tok in ("NaN", "Infinity", '"undefined"'):
    if tok in raw:
        bad.append(f"literal {tok} present in results.json")

print("\nfeatured clips:")
for f in D["featured"]:
    mb = os.path.getsize(os.path.join("docs", "clips", "featured", f["mp4"])) / 1e6
    print(f"  {f['task']:22s} {f['duration']:6.1f}s  {len(f['segments']):3d} spans  "
          f"{len(f['trace_speed']):5d} trace pts  {mb:5.2f} MB  "
          f"outliers={f['flags']['n_outlier']} idle={f['flags']['idle_frac']}")

print("\nheadline numbers:")
print(f"  corpus            {D['corpus']['episodes']:,} episodes / {D['corpus']['hours']:,} h")
print(f"  audited           {D['prevalence']['annotated_episodes']:,}")
print(f"  flagged           {D['prevalence']['flagged']:,} ({D['prevalence']['flagged_pct']}%)")
print(f"  wasted runtime    {D['prevalence']['lost_hours']} h")
print(f"  efficiency penalty {D['validation']['efficiency_penalty']}x")
print(f"  unlabelled active {D['validation']['unlabelled']['active_pct']}% "
      f"({D['validation']['unlabelled']['active_hours']} h)")

print("\n" + ("PROBLEMS:\n  " + "\n  ".join(bad) if bad else "OK - no problems found"))
