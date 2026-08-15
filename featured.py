"""Pull preview MP4 + per-frame hand-speed trace for the demo episodes."""
import json

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("boto3", "pandas", "pyarrow", "zarr==3.1.5", "numpy", "awscli")
)

app = modal.App("failure-radar-featured")
vol = modal.Volume.from_name("egoverse-cache")

FEATURED = [
    "696d023ba534b5c917588ab1",  # sorting_coffee_beans   24.1x outlier
    "696b9f9a5fedf8533255dcb0",  # peeling_potatoes       12.9x outlier
    "69b4aea25fa639c4e24e740b",  # cleaning_bags          15.2x outlier
    "696bd3d59a140887948c6044",  # packaging_food         11.6x outlier
    "69b4b1bc164aaac0a2a6ec44",  # packaging_nuts         10.6x outlier
    "696cd612804859de93b59a99",  # folding_boxes          10.4x outlier
    "696cd9c76ed3eb90778f0657",  # peeling_garlic         100% unlabelled
    "696da165bad2864f69af2c3d",  # assembling_boxes       100% unlabelled
    "696bba6714764da1c7e184bf",  # soldering_batteries    100% unlabelled
]


def load_r2_env():
    import os
    import shutil
    import subprocess

    cached = "/cache/egoverse_env"
    if not os.path.exists(cached):
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/GaTech-RL2/EgoVerse.git", "/tmp/EgoVerse"],
            check=True, capture_output=True)
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


@app.function(image=image, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("egoverse-aws")], timeout=1800)
def build_featured():
    import os
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    import boto3
    import numpy as np
    import pandas as pd

    env = load_r2_env()
    bucket = env.get("BUCKET", "rldb")
    s3 = boto3.client("s3", endpoint_url=env["AWS_ENDPOINT_URL_S3"],
                      aws_access_key_id=env["R2_ACCESS_KEY_ID"],
                      aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
                      region_name="auto")

    df = pd.read_parquet("/cache/episodes.parquet").set_index("episode_hash")
    outdir = "/cache/featured"
    os.makedirs(outdir, exist_ok=True)

    manifest = []
    for h in FEATURED:
        if h not in df.index:
            print("MISSING from table:", h)
            continue
        row = df.loc[h]
        prefix = row["zarr_processed_path"].replace(f"s3://{bucket}/", "").rstrip("/")

        # ---- pose arrays ----
        local = f"/tmp/{h}.zarr"
        shutil.rmtree(local, ignore_errors=True)
        os.makedirs(local, exist_ok=True)
        keys, token = [], None
        while True:
            kw = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
            if token:
                kw["ContinuationToken"] = token
            r = s3.list_objects_v2(**kw)
            for o in r.get("Contents", []):
                rel = o["Key"][len(prefix):].lstrip("/")
                if rel == "zarr.json" or rel.startswith(
                        ("left.obs_ee_pose/", "right.obs_ee_pose/")):
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
                a = np.asarray(store[k][:])[:T, :3]
                hands[side] = a
        n = min(len(v) for v in hands.values())
        bad = np.zeros(n, bool)
        for v in hands.values():
            bad |= (np.abs(v[:n]) > 1e8).any(axis=-1)

        sp = []
        for side, xyz in hands.items():
            d = np.linalg.norm(np.diff(xyz[:n], axis=0), axis=-1)
            sp.append(np.concatenate([[0.0], d]))
        speed = np.nanmax(np.stack(sp), axis=0)
        speed[bad] = np.nan

        # smooth + downsample to ~10 Hz for the UI
        win = max(1, int(fps // 3))
        kern = np.ones(win) / win
        filled = np.nan_to_num(speed, nan=0.0)
        smooth = np.convolve(filled, kern, mode="same")
        step = max(1, int(round(fps / 10)))
        trace = smooth[::step]
        times = (np.arange(len(smooth))[::step] / fps)

        # ---- preview mp4 ----
        mp4_key = row["zarr_mp4_path"].replace(f"s3://{bucket}/", "")
        mp4_name = f"{h}.mp4"
        try:
            s3.download_file(bucket, mp4_key, os.path.join(outdir, mp4_name))
            mp4_ok = True
        except Exception as e:
            print("mp4 failed", h, e)
            mp4_ok = False

        segs = []
        for s in (row["segments"] if row["segments"] is not None else []):
            segs.append({
                "label": s.get("label"),
                "start": s.get("start_seconds"),
                "end": s.get("end_seconds"),
            })

        manifest.append({
            "episode_hash": h,
            "task": row["task"],
            "lab": row["lab"],
            "total_frames": T,
            "fps": fps,
            "duration": n / fps,
            "sentinel_frac": float(bad.mean()),
            "mp4": mp4_name if mp4_ok else None,
            "segments": segs,
            "trace_hz": fps / step,
            "trace_t": [round(float(x), 2) for x in times],
            "trace_speed": [round(float(x), 6) for x in trace],
        })
        print("built", h, row["task"], f"{n / fps:.1f}s", "mp4=", mp4_ok)
        shutil.rmtree(local, ignore_errors=True)

    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh)
    vol.commit()
    return {"built": len(manifest)}


@app.local_entrypoint()
def main():
    print(json.dumps(build_featured.remote(), indent=2))
