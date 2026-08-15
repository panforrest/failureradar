# FailureRadar

**Which EgoVerse demonstrations teach a robot the wrong thing — and where, to the second.**

Robotics Hackathon, 15 Aug 2026 · Track 3, The Human Reward Model · built on the
[EgoVerse](https://github.com/GaTech-RL2/EgoVerse) dataset with Modal compute.

| Asset | Link |
|---|---|
| Live dashboard | https://panforrest.github.io/failureradar/ |
| Demo video | https://panforrest.github.io/failureradar/FailureRadar_demo.mp4 |
| Repo | https://github.com/panforrest/failureradar |

---

## The finding

EgoVerse has no ground-truth success labels. `is_eval` is `False` for all 446,957
episodes, `eval_success` is `True` for 428,310 of them and `False` for exactly 6,
and `eval_score` is `-1` almost everywhere. Nothing in the dataset tells you which
demonstrations are good.

So FailureRadar infers it, from two things that are already there: **the shape of the
annotation timeline**, and **the geometry of the hands**.

| | |
|---|---|
| Episodes indexed | 446,957 (4,002.7 h) |
| Episodes audited for demonstration quality | 23,289 |
| Flagged | 753 (3.23%) |
| Runtime that teaches a policy nothing | 8.67 h |
| Flagged spans' path efficiency vs. duration-matched normal spans | **1.54× worse**, p = 1.6×10⁻¹⁴ |
| Null-labelled spans that actually contain active manipulation | **60.6%** — 1.62 h of recoverable data in the sample alone |

---

## How it works

### 1. The detector that didn't work

The obvious idea is that humans repeat an action when they fail, so repeated
annotation labels should indicate failure. The top-ranked episode under that rule
was a person scooping coffee beans 66 times — a perfectly good demonstration.
**Repetition is a property of the task, not of failure.** That approach is a dead end
and the repo keeps it in `segments_probe.py` as the negative result it is.

### 2. Cycle-time outliers

Normalise *within* the episode instead. For every label repeated ≥3 times, take the
median duration of that label in that episode, and flag spans running ≥3× that median
and ≥3 s absolute.

A repetitive task now supplies its own tight baseline, so the more cyclic the work,
the *more* sensitive the detector becomes. In `peeling_potatoes`, peeling spans run
2.7–5.5 s — except one that runs **60.1 s**. In `cleaning_bags`, "apply polish on
sponge" runs 1.0–2.6 s — except one at **39.4 s**.

### 3. Null-labelled spans

Some episodes carry segment timings with no label text at all — 9 spans across 100
seconds, every one `None`. Rather than assume these are pauses, we measured them.
They are not idle: 60.6% move faster than the 25th percentile of *labelled* work.
These are annotation failures, and the data is recoverable by re-annotation rather
than something to drop.

### 4. Kinematic corroboration — the part that makes it defensible

The flags come from annotation timestamps. The check comes from geometry, and the two
never touch.

For 800 episodes we downloaded **only** `left/right.obs_ee_pose` — 4.4 MB out of a
271 MB store, 1.6%, skipping every JPEG frame — on 60 parallel Modal containers, then
computed per-span hand speed, dwell fraction and **path efficiency** (straight-line
displacement ÷ path length; low efficiency means the hand moved a lot and arrived
nowhere).

Flagged spans are 3.3× less efficient than same-label baseline spans. But efficiency
falls with duration by construction and flagged spans are long by definition, so that
number alone proves nothing. Controlling for it:

* Compared only against **non-flagged spans of comparable length**, flagged spans are
  less efficient in **all seven** duration bins (ratios 0.59–0.76).
* Against a trend fitted on non-flagged spans, `log(eff) = -0.761·log(dur) - 1.060`,
  flagged spans sit **1.54× below what duration alone predicts** (Mann-Whitney
  p = 1.6×10⁻¹⁴).

A flag derived purely from annotation timing predicts independently measured motion
inefficiency. That is the whole claim.

---

## Corpus contract audit

Separately, every one of the 446,957 index rows is checked against the rules in
EgoVerse's own `CONTRIBUTING_DATA.md`. No download needed, so this covers the entire
corpus rather than a sample. Selected findings:

* **355,927 episodes have no operator attribution.** §5.1 requires a hashed operator
  ID. Without one you cannot detect operator-specific bias or exclude a single
  demonstrator's data.
* **43,271 episodes have an operator field that is not a hash.** The guide says
  operator "MUST be hashed before insertion — never store raw names/emails". The
  table contains plain strings including `ABC`, `Baoyu`, `scale`.
* **386,059 episodes carry no license field**, while the consortium redistributes
  under CC BY-SA 4.0.
* **23,013 distinct task strings from one contributor.** Stemming collapses the
  corpus's task vocabulary substantially — `prepare_onion`/`prepare_onions`,
  `iron_clothes`/`ironing_clothes`. The guide asks for canonical categories, "not a
  one-off trial description". Fragmented task names break every per-task sampling or
  balancing strategy.
* **Episodes with `num_frames` as low as -2**, and thousands under 5 seconds.
* Only mecka ships segment annotations, so every other contributor is currently
  **unauditable** for demonstration quality.

---

## Deliverables

1. **Tagged episodes** — every flagged episode with per-span reasons.
2. **Prevalence audit** — rates by signal and by task, plus the corpus contract audit.
3. **Confidence meter over video** — preview MP4 with a per-frame hand-speed strip,
   flagged spans shaded, click-to-seek.
4. **A drop-list you can actually use.** The dashboard exports a `DatasetFilter` that
   plugs straight into EgoVerse's own `sync_s3.py`:

```python
DATA_FILTERS["failureradar-clean"] = DatasetFilter(
    filter_lambdas=[
        "lambda row: row.get('episode_hash') not in FAILURERADAR_DROP",
        "lambda row: not row.get('is_deleted')",
        "lambda row: (row.get('num_frames') or 0) >= 150",
    ]
)
```

---

## Running it

```bash
pip install modal pandas pyarrow numpy
modal setup
modal volume create egoverse-cache
modal secret create egoverse-aws AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=us-east-2

modal run setup_db.py       # pull app.episodes -> parquet on the volume
modal run scan.py           # 800 episodes, 60 containers, pose arrays only
modal run featured.py       # preview MP4s + per-frame speed traces
python build_results.py     # -> docs/results.json
python -m http.server 8777 --directory docs
```

You do **not** need to install the EgoVerse package. Its `requirements.txt` pins
`mujoco-py==2.1.2.14`, `dm-control`, `projectaria-tools` and `torch`, none of which
are needed to read the data. The Zarr v3 schema is fully documented in
`CONTRIBUTING_DATA.md`, so `zarr==3.1.5` + `numpy` is enough.

### Two things in the format that will silently corrupt your results

* **Arrays are zero-padded past `total_frames`** to a 100-frame chunk boundary. Slice
  `[:total_frames]` or every episode appears to end with the hand teleporting to the
  origin.
* **Missing keypoints use a `1e9` sentinel, not `NaN`.** `np.isnan()` won't catch it,
  and one unmasked dropout produces a velocity spike of 10⁹ that swamps every
  statistic downstream.

---

## How Modal is used

* **Fan-out scan** — `scan_episode.map()` across 60 containers; 800 episodes, 7,518
  spans, zero failures, ~3 minutes. Serially this is over an hour.
* **Volume** — `egoverse-cache` holds the episode table, the R2 credentials, scan
  results and featured clips, so reruns are instant and no work is repeated.
* **Secrets** — R2/AWS credentials injected rather than baked into the image.
* **Selective download** — listing each Zarr store and fetching only the pose arrays
  cuts per-episode transfer by 60×, which is what makes an 800-episode scan finish in
  minutes on hackathon wifi.

---

## Honest limitations

* **Scope.** Prevalence figures describe the 23,289 annotated episodes, not all
  446,957. Only mecka publishes segment annotations.
* **No ground truth.** Nothing here is calibrated against human success/failure
  judgements, because EgoVerse contains none. What is measured is *runtime that
  teaches a policy nothing* — stalls, rework, and unlabelled work — which is a
  narrower and more defensible claim than "this demo failed".
* **Sampling.** The kinematic validation used 400 flagged + 400 clean episodes, a
  deliberately balanced sample. The duration-controlled comparison is within-sample
  and does not assume the base rate.
* **Episode-level averages wash the signal out.** The effect lives at span level,
  which is why the deliverable is a confidence meter over a segment rather than a
  single per-episode score.

---

## Layout

```
failureradar.py      scoring core - idle, cycle-time outliers, coverage, struggle language
setup_db.py          Modal: pull app.episodes to parquet
scan.py              Modal: fan-out kinematic scan (pose arrays only)
featured.py          Modal: preview MP4s + per-frame speed traces
build_results.py     assemble docs/results.json
validate.py          flagged vs same-label baseline
validate2.py         duration-controlled validation + residual test
segments_probe.py    the repetition detector that failed, kept as a negative result
record.js            render the narrated demo video from the live dashboard
docs/index.html      dashboard
```
