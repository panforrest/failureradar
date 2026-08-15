"""FailureRadar kinematic scan.

For a sample of mecka episodes, download ONLY the end-effector pose arrays
(1.6% of each zarr store) and compute per-annotation-segment motion features.
Used to test whether annotation-derived anomaly flags are corroborated by
independent kinematic evidence.
"""
import json

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("boto3", "pandas", "pyarrow", "zarr==3.1.5", "numpy", "awscli")
)

app = modal.App("failure-radar-scan")
vol = modal.Volume.from_name("egoverse-cache")

STILL_THRESH = 0.002  # m/frame; below this the hand is effectively stationary


def load_r2_env():
    import os
    import shutil
    import subprocess

    cached = "/cache/egoverse_env"
    if not os.path.exists(cached):
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/GaTech-RL2/EgoVerse.git", "/tmp/EgoVerse"],
            check=True, capture_output=True,
        )
        subprocess.run(["bash", "egomimic/utils/aws/setup_secret.sh"],
                       cwd="/tmp/EgoVerse", check=True, capture_output=True)
        shutil.copy(os.path.expanduser("~/.egoverse_env"), cached)
        vol.commit()

    env = {}
    for line in open(cached):
        line = line.strip().removeprefix("export ")
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k] = v.strip().strip('"').strip("'")
    return env


def s3_client(env):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=env.get("AWS_ENDPOINT_URL_S3"),
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


@app.function(image=image, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("egoverse-aws")],
              timeout=900, max_containers=60, retries=1)
def scan_episode(job: dict):
    """Download ee_pose arrays for one episode; return per-segment kinematics."""
    import os
    import shutil
    import traceback
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    h = job["episode_hash"]
    try:
        env = load_r2_env()
        s3 = s3_client(env)
        bucket = env.get("BUCKET", "rldb")
        prefix = job["zarr_uri"].replace(f"s3://{bucket}/", "").rstrip("/")
        local = f"/tmp/{h}.zarr"
        shutil.rmtree(local, ignore_errors=True)
        os.makedirs(local, exist_ok=True)

        wanted = ("zarr.json", "left.obs_ee_pose/", "right.obs_ee_pose/")
        keys, token = [], None
        while True:
            kw = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
            if token:
                kw["ContinuationToken"] = token
            r = s3.list_objects_v2(**kw)
            for o in r.get("Contents", []):
                rel = o["Key"][len(prefix):].lstrip("/")
                if rel == "zarr.json" or rel.startswith(("left.obs_ee_pose/",
                                                        "right.obs_ee_pose/")):
                    keys.append((o["Key"], rel))
            if not r.get("IsTruncated"):
                break
            token = r["NextContinuationToken"]

        def get(item):
            key, rel = item
            dst = os.path.join(local, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            s3.download_file(bucket, key, dst)

        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(get, keys))

        import zarr

        store = zarr.open(local, mode="r")
        attrs = dict(store.attrs)
        T = int(attrs.get("total_frames") or 0)
        fps = float(attrs.get("fps") or 30.0)

        hands = {}
        for side in ("left", "right"):
            k = f"{side}.obs_ee_pose"
            if k in store:
                a = np.asarray(store[k][:])
                if T:
                    a = a[:T]
                hands[side] = a[:, :3]
        if not hands:
            return {"episode_hash": h, "error": "no ee_pose"}

        # sentinel mask (1e9 = missing/degenerate tracking, per CONTRIBUTING_DATA)
        n_frames = min(len(v) for v in hands.values())
        bad = np.zeros(n_frames, dtype=bool)
        for v in hands.values():
            bad |= (np.abs(v[:n_frames]) > 1e8).any(axis=-1)

        speeds = {}
        for side, xyz in hands.items():
            xyz = xyz[:n_frames].copy()
            d = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
            d = np.concatenate([[0.0], d])
            d[bad] = np.nan
            speeds[side] = d
        combined = np.nanmax(np.stack(list(speeds.values())), axis=0)

        out_segs = []
        for seg in job["segments"]:
            a = seg.get("start_seconds")
            b = seg.get("end_seconds")
            if a is None or b is None:
                continue
            i0 = max(0, int(float(a) * fps))
            i1 = min(n_frames, int(float(b) * fps))
            if i1 - i0 < 2:
                continue
            w = combined[i0:i1]
            valid = w[~np.isnan(w)]
            if len(valid) < 2:
                continue
            xyz = {s: hands[s][i0:i1] for s in hands}
            path_len = float(np.nansum(valid))
            disp = float(max(
                np.linalg.norm(v[-1] - v[0]) for v in xyz.values()
            ))
            out_segs.append({
                "label": (seg.get("label") or "").strip().lower(),
                "start": float(a),
                "end": float(b),
                "dur": float(b) - float(a),
                "n_frames": int(i1 - i0),
                "mean_speed": float(np.nanmean(valid)),
                "med_speed": float(np.nanmedian(valid)),
                "p95_speed": float(np.nanpercentile(valid, 95)),
                "dwell_frac": float((valid < STILL_THRESH).mean()),
                "path_len": path_len,
                "disp": disp,
                "efficiency": float(disp / path_len) if path_len > 1e-6 else 0.0,
            })

        shutil.rmtree(local, ignore_errors=True)
        return {
            "episode_hash": h,
            "lab": job["lab"],
            "task": job["task"],
            "total_frames": T,
            "fps": fps,
            "sentinel_frac": float(bad.mean()),
            "n_seg": len(out_segs),
            "segments": out_segs,
            "error": "",
        }
    except Exception:
        return {"episode_hash": h, "error": traceback.format_exc()[-800:]}


@app.function(image=image, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("egoverse-aws")], timeout=3600)
def run_scan(n_flagged: int = 400, n_control: int = 400, seed: int = 0):
    import re
    import statistics as st
    from collections import defaultdict

    import numpy as np
    import pandas as pd

    df = pd.read_parquet("/cache/episodes.parquet")
    df = df[~df["is_deleted"]]
    df = df[df["segments"].map(lambda v: isinstance(v, np.ndarray) and len(v) > 0)]
    df = df[df["zarr_processed_path"] != ""]
    print("candidate episodes:", len(df))

    IDLE = {"none", "no action", "", "idle", "nothing"}

    def norm(s):
        s = str(s).lower().strip()
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s))

    def flags(segs):
        items = []
        for s in segs:
            lab = norm(s.get("label", ""))
            try:
                a, b = float(s.get("start_seconds")), float(s.get("end_seconds"))
            except (TypeError, ValueError):
                continue
            items.append((lab, a, b, max(0.0, b - a)))
        if not items:
            return 0, 0.0
        span = max(b for _, _, b, _ in items)
        idle = sum(d for lab, _, _, d in items if lab in IDLE)
        by = defaultdict(list)
        for lab, _, _, d in items:
            by[lab].append(d)
        n_out = 0
        for lab, ds in by.items():
            if len(ds) >= 3 and lab not in IDLE:
                med = st.median(ds)
                if med > 0:
                    n_out += sum(1 for d in ds if d / med >= 3.0 and d >= 3.0)
        return n_out, (idle / span if span > 0 else 0.0)

    stats = [flags(s) for s in df["segments"]]
    df["n_outlier"] = [x[0] for x in stats]
    df["idle_frac"] = [x[1] for x in stats]

    flagged = df[(df["n_outlier"] >= 1) | (df["idle_frac"] > 0.10)]
    clean = df[(df["n_outlier"] == 0) & (df["idle_frac"] == 0)]
    print(f"flagged pool={len(flagged)}  clean pool={len(clean)}")

    fl = flagged.sample(min(n_flagged, len(flagged)), random_state=seed)
    cl = clean.sample(min(n_control, len(clean)), random_state=seed)
    sample = pd.concat([fl, cl])
    print("scanning", len(sample), "episodes")

    jobs = [
        {
            "episode_hash": r["episode_hash"],
            "zarr_uri": r["zarr_processed_path"],
            "lab": r["lab"],
            "task": r["task"],
            "segments": [
                {
                    "label": s.get("label"),
                    "start_seconds": s.get("start_seconds"),
                    "end_seconds": s.get("end_seconds"),
                }
                for s in r["segments"]
            ],
        }
        for _, r in sample.iterrows()
    ]

    results = list(scan_episode.map(jobs, return_exceptions=True))
    ok = [r for r in results if isinstance(r, dict) and not r.get("error")]
    bad = [r for r in results if not isinstance(r, dict) or r.get("error")]
    print(f"OK={len(ok)}  FAILED={len(bad)}")
    for r in bad[:3]:
        print("sample failure:", str(r)[:600])

    rows = []
    for r in ok:
        for s in r["segments"]:
            rows.append({
                "episode_hash": r["episode_hash"],
                "task": r["task"],
                "sentinel_frac": r["sentinel_frac"],
                **s,
            })
    seg = pd.DataFrame(rows)
    seg.to_parquet("/cache/kinematics_segments.parquet")

    meta = sample[["episode_hash", "task", "lab", "num_frames",
                   "n_outlier", "idle_frac", "zarr_mp4_path",
                   "zarr_processed_path"]].copy()
    meta["scanned"] = meta["episode_hash"].isin({r["episode_hash"] for r in ok})
    meta.to_parquet("/cache/scan_meta.parquet")
    vol.commit()

    print("\nsegments captured:", len(seg))
    print(seg[["dur", "mean_speed", "dwell_frac", "efficiency"]].describe())
    return {"ok": len(ok), "failed": len(bad), "segments": len(seg)}


@app.local_entrypoint()
def main(n_flagged: int = 400, n_control: int = 400):
    print(json.dumps(run_scan.remote(n_flagged, n_control), indent=2))
