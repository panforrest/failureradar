"""Verify we can pull preview MP4s and numeric zarr arrays out of R2."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("boto3", "pandas", "pyarrow", "zarr==3.1.5", "numpy", "awscli")
)

app = modal.App("failure-radar-probe")
vol = modal.Volume.from_name("egoverse-cache")


def load_r2_env():
    """Run setup_secret.sh once, cache the resulting env on the volume."""
    import os
    import subprocess

    cached = "/cache/egoverse_env"
    if not os.path.exists(cached):
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/GaTech-RL2/EgoVerse.git", "/tmp/EgoVerse"],
            check=True, capture_output=True,
        )
        r = subprocess.run(["bash", "egomimic/utils/aws/setup_secret.sh"],
                           cwd="/tmp/EgoVerse", capture_output=True, text=True)
        print("setup_secret rc:", r.returncode)
        print("stdout:", r.stdout[-3000:])
        print("stderr:", r.stderr[-3000:])
        src = os.path.expanduser("~/.egoverse_env")
        if not os.path.exists(src):
            raise RuntimeError("setup_secret.sh did not produce ~/.egoverse_env")
        import shutil
        shutil.copy(src, cached)
        vol.commit()

    env = {}
    for line in open(cached):
        line = line.strip().removeprefix("export ")
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k] = v.strip().strip('"').strip("'")
            os.environ[k] = env[k]
    return env


@app.function(image=image, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("egoverse-aws")], timeout=1200)
def probe():
    import io
    import json
    import time

    import boto3
    import numpy as np
    import pandas as pd

    env = load_r2_env()
    print("R2 env keys:", sorted(env.keys()))
    endpoint = env.get("AWS_ENDPOINT_URL_S3") or env.get("R2_ENDPOINT_URL")
    bucket = env.get("BUCKET", "rldb")
    print("endpoint:", endpoint, "bucket:", bucket)

    def make_client(with_token):
        kw = dict(
            endpoint_url=endpoint,
            aws_access_key_id=env["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        if with_token and env.get("R2_SESSION_TOKEN"):
            kw["aws_session_token"] = env["R2_SESSION_TOKEN"]
        return boto3.client("s3", **kw)

    s3 = None
    for with_token in (False, True):
        c = make_client(with_token)
        try:
            c.list_objects_v2(Bucket=bucket, Prefix="processed_v3/", MaxKeys=1)
            print(f"auth OK with_token={with_token}")
            s3 = c
            break
        except Exception as e:
            print(f"auth FAILED with_token={with_token}: {type(e).__name__}: {e}")
    if s3 is None:
        return "AUTH FAILED"

    df = pd.read_parquet("/cache/episodes.parquet")
    row = df[(df["lab"] == "mecka") & (df["zarr_mp4_path"] != "")].iloc[0]
    h = row["episode_hash"]
    zarr_uri = row["zarr_processed_path"]
    mp4_uri = row["zarr_mp4_path"]
    print("\nepisode:", h)
    print("zarr:", zarr_uri)
    print("mp4 :", mp4_uri)

    prefix = zarr_uri.replace(f"s3://{bucket}/", "").rstrip("/")

    # 1) list what's inside the zarr store
    t0 = time.time()
    keys = []
    token = None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        keys += [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            break
        token = r["NextContinuationToken"]
    print(f"\nlisted {len(keys)} objects in {time.time() - t0:.1f}s")

    total = sum(s for _, s in keys)
    print(f"total store size: {total / 1e6:.1f} MB")
    groups = {}
    for k, s in keys:
        g = k[len(prefix):].lstrip("/").split("/")[0]
        groups[g] = groups.get(g, [0, 0])
        groups[g][0] += 1
        groups[g][1] += s
    print("\nper-array bytes:")
    for g, (n, s) in sorted(groups.items(), key=lambda x: -x[1][1]):
        print(f"  {g:32s} {n:>5d} objs  {s / 1e6:9.2f} MB")

    numeric = [g for g in groups if not g.startswith("images.")]
    num_bytes = sum(groups[g][1] for g in numeric)
    print(f"\nNUMERIC-ONLY total: {num_bytes / 1e6:.2f} MB "
          f"({100 * num_bytes / total:.1f}% of store)")

    # 2) read zarr.json attrs
    obj = s3.get_object(Bucket=bucket, Key=f"{prefix}/zarr.json")
    meta = json.loads(obj["Body"].read())
    attrs = meta.get("attributes", meta)
    print("\nzarr.json attrs keys:", list(attrs.keys())[:20])
    print("total_frames:", attrs.get("total_frames"), "fps:", attrs.get("fps"))
    print("embodiment:", attrs.get("embodiment"))
    print("features:", list((attrs.get("features") or {}).keys()))

    # 3) download the numeric arrays only, open with zarr
    import os
    local = f"/tmp/{h}.zarr"
    os.makedirs(local, exist_ok=True)
    t0 = time.time()
    n_files = 0
    for k, s in keys:
        rel = k[len(prefix):].lstrip("/")
        if rel.split("/")[0].startswith("images."):
            continue
        dst = os.path.join(local, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        s3.download_file(bucket, k, dst)
        n_files += 1
    dt = time.time() - t0
    print(f"\ndownloaded {n_files} numeric objects in {dt:.1f}s "
          f"({num_bytes / 1e6 / max(dt, 0.01):.1f} MB/s)")

    import zarr
    st = zarr.open(local, mode="r")
    print("\nzarr keys:", list(st.keys()))
    T = attrs.get("total_frames")
    for key in ["left.obs_ee_pose", "right.obs_ee_pose", "obs_head_pose"]:
        if key in st:
            a = st[key][:]
            print(f"{key}: shape={a.shape} dtype={a.dtype}")
            xyz = a[:T, :3] if T else a[:, :3]
            sentinel = (np.abs(xyz) > 1e8).any(axis=-1)
            print(f"   sentinel(1e9) frames: {int(sentinel.sum())}/{len(xyz)}")
            good = xyz[~sentinel]
            if len(good) > 1:
                v = np.linalg.norm(np.diff(good, axis=0), axis=-1)
                print(f"   xyz range: {good.min(axis=0).round(3)} .. {good.max(axis=0).round(3)}")
                print(f"   step median={np.median(v):.5f} m  p99={np.percentile(v, 99):.5f} m")

    # 4) download the preview mp4
    mp4_key = mp4_uri.replace(f"s3://{bucket}/", "")
    t0 = time.time()
    buf = io.BytesIO()
    s3.download_fileobj(bucket, mp4_key, buf)
    print(f"\nMP4 {mp4_key}: {buf.tell() / 1e6:.2f} MB in {time.time() - t0:.1f}s")

    return "OK"


@app.function(image=image, volumes={"/cache": vol},
              secrets=[modal.Secret.from_name("egoverse-aws")], timeout=1200)
def probe_safe():
    import traceback
    try:
        return probe.local()
    except Exception:
        return "EXCEPTION:\n" + traceback.format_exc()


@app.local_entrypoint()
def main():
    print(probe_safe.remote())
