"""Render demo/narration.md to per-segment MP3s via ElevenLabs.

The API key is read from the environment or a local .env file, both of which are
gitignored. It is never printed and never written into the repo.

    python make_vo.py --list          # show voices on the account
    python make_vo.py                 # render every segment
    python make_vo.py --only 04       # re-render one segment
"""
import argparse
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

API = "https://api.elevenlabs.io/v1"
SCRIPT = pathlib.Path("demo/narration.md")
OUTDIR = pathlib.Path("demo/vo")

# Long-form narration model: 10k char limit, steadier than the expressive v3.
MODEL = "eleven_multilingual_v2"
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"


def api_key():
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        env = pathlib.Path(".env")
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip().removeprefix("export ")
                if line.startswith("ELEVENLABS_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit(
            "No API key found.\n"
            "Create a file named .env in this folder containing:\n"
            "    ELEVENLABS_API_KEY=your_key_here\n"
            "(.env is gitignored and will not be committed.)"
        )
    return key


def request(path, key, data=None, raw=False):
    req = urllib.request.Request(
        API + path,
        data=data,
        headers={"xi-api-key": key,
                 **({"Content-Type": "application/json"} if data else {})},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.read() if raw else r.read().decode()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:400]
        sys.exit(f"ElevenLabs {e.code} on {path}\n{body}")


def list_voices(key):
    import json
    v = json.loads(request("/voices", key))["voices"]
    print(f"{len(v)} voices on this account:\n")
    for x in v:
        labels = x.get("labels") or {}
        desc = ", ".join(f"{k}={val}" for k, val in list(labels.items())[:3])
        print(f"  {x['voice_id']}  {x['name']:<22} {desc}")
    print("\nPass one with --voice <id>.")


def parse_segments(only=None):
    text = SCRIPT.read_text(encoding="utf-8")
    segs = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        head, *rest = block.split("\n", 1)
        name = head.split("—")[0].strip()
        body = rest[0] if rest else ""
        # drop the [on screen] cue lines and any code fences
        lines = [ln for ln in body.splitlines()
                 if not ln.strip().startswith("`[on screen]")
                 and not ln.strip().startswith("---")]
        spoken = "\n".join(lines).strip()
        spoken = re.sub(r"\n{3,}", "\n\n", spoken)
        if spoken and (only is None or only in name):
            segs.append((name, spoken))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--only")
    a = ap.parse_args()

    key = api_key()
    if a.list:
        return list_voices(key)

    import json
    OUTDIR.mkdir(parents=True, exist_ok=True)
    segs = parse_segments(a.only)
    if not segs:
        sys.exit("no segments matched")

    total_chars = 0
    for name, text in segs:
        payload = json.dumps({
            "text": text,
            "model_id": a.model,
            "voice_settings": {"stability": 0.45, "similarity_boost": 0.75,
                               "style": 0.0, "use_speaker_boost": True},
        }).encode()
        audio = request(
            f"/text-to-speech/{a.voice}?output_format=mp3_44100_128",
            key, data=payload, raw=True)
        dst = OUTDIR / f"{name}.mp3"
        dst.write_bytes(audio)
        words = len(text.split())
        total_chars += len(text)
        print(f"{name:<16} {words:>4} words  ~{words / 2.5:5.0f}s  "
              f"{len(audio) / 1e6:.2f} MB  -> {dst}")

    print(f"\n{len(segs)} segments, {total_chars} characters billed.")
    print(f"Audio in {OUTDIR}/ (gitignored).")


if __name__ == "__main__":
    main()
