import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "unzip")
    .pip_install("boto3", "sqlalchemy", "psycopg[binary]", "pandas", "pyarrow", "awscli")
)

app = modal.App("failure-radar")
vol = modal.Volume.from_name("egoverse-cache")


@app.function(
    image=image,
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("egoverse-aws")],
    timeout=1800,
)
def fetch_table():
    import json
    import os
    import subprocess

    import boto3
    import pandas as pd
    from sqlalchemy import URL, MetaData, Table, create_engine, select

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/GaTech-RL2/EgoVerse.git",
            "/tmp/EgoVerse",
        ],
        check=True,
    )
    r = subprocess.run(
        ["bash", "egomimic/utils/aws/setup_secret.sh"],
        cwd="/tmp/EgoVerse",
        capture_output=True,
        text=True,
    )
    print("setup_secret stdout:", r.stdout[-2000:])
    print("setup_secret stderr:", r.stderr[-2000:])

    env_path = os.path.expanduser("~/.egoverse_env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip().removeprefix("export ")
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k] = v.strip().strip('"').strip("'")
                print("env:", k)
    else:
        print("NO ~/.egoverse_env was written")

    sm = boto3.client(
        "secretsmanager", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-2")
    )
    arn = os.environ.get("SECRETS_ARN")
    if not arn:
        try:
            arns = [s["ARN"] for s in sm.list_secrets()["SecretList"]]
            print("visible secrets:", arns)
            arn = arns[0] if arns else None
        except Exception as e:
            print("list_secrets failed:", e)
    if not arn:
        raise SystemExit("could not resolve SECRETS_ARN")

    cfg = json.loads(sm.get_secret_value(SecretId=arn)["SecretString"])
    engine = create_engine(
        URL.create(
            "postgresql+psycopg",
            username=cfg.get("username") or cfg.get("user"),
            password=cfg["password"],
            host=cfg["host"],
            port=cfg.get("port", 5432),
            database=cfg.get("dbname", "appdb"),
            query={"sslmode": "require"},
        )
    )

    t = Table("episodes", MetaData(), autoload_with=engine, schema="app")
    with engine.connect() as c:
        df = pd.DataFrame(
            c.execute(select(t)).fetchall(), columns=[col.name for col in t.columns]
        )

    df.to_parquet("/cache/episodes.parquet")
    vol.commit()

    print("EPISODES:", len(df))
    print(df.groupby("task").size().sort_values(ascending=False).head(20))
    print(df.groupby("lab").size())
    print(df.columns.tolist())
    print(
        df[
            [
                "episode_hash",
                "task",
                "lab",
                "num_frames",
                "zarr_processed_path",
                "zarr_mp4_path",
            ]
        ]
        .head()
        .to_string()
    )
    return len(df)


@app.local_entrypoint()
def main():
    print("rows:", fetch_table.remote())
