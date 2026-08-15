"""FailureRadar scoring core.

Signals are derived from annotation structure only (no LLM, no video decode):

  idle_frac     fraction of episode wall-time covered by null/"no action" labels
  cycle_outlier a segment whose duration is >=3x the median duration of the SAME
                label within the SAME episode (and >=3s absolute). Normalising
                per-label-per-episode means legitimately repetitive tasks are not
                penalised - they supply the tightest baseline.
  coverage      fraction of episode wall-time covered by any annotation span
  struggle      explicit recovery language in a label
"""
import re
import statistics as st
from collections import defaultdict

IDLE_LABELS = {"none", "no action", "", "idle", "nothing", "no motion", "null"}

# Deliberately narrow: earlier broad patterns matched "paper slip" and
# "fix ring on tag", which are ordinary actions rather than recoveries.
STRUGGLE_RE = re.compile(
    r"\b(again|retry|re-?attempt|second attempt|missed|fail(ed|s|ure)?|"
    r"slipped|slips out|fumbl\w*|knock(ed|s)? over|spill(ed|s)?|"
    r"re-?adjust|re-?do|accidental\w*|drop(s|ped)? (it |them )?(on|onto) "
    r"(the )?(floor|ground)|pick(s|ed)? (it |them )?back up)\b"
)

OUTLIER_RATIO = 3.0
OUTLIER_MIN_SEC = 3.0
MIN_REPEATS = 3


def norm_label(s):
    s = str(s or "").lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_segments(segs):
    out = []
    for s in segs:
        a, b = _f(s.get("start_seconds")), _f(s.get("end_seconds"))
        if a is None or b is None or b < a:
            continue
        raw = s.get("label")
        out.append({
            "raw": (raw if raw is not None else "None"),
            "label": norm_label(raw),
            "start": a,
            "end": b,
            "dur": b - a,
        })
    out.sort(key=lambda d: d["start"])
    return out


def score_episode(segs):
    """Return (episode_features, per_segment_flags)."""
    items = parse_segments(segs)
    n = len(items)
    if n == 0:
        return {
            "n_seg": 0, "span": 0.0, "coverage": 0.0, "idle_frac": 0.0,
            "n_outlier": 0, "worst_ratio": 0.0, "n_struggle": 0,
            "lost_sec": 0.0, "flag": True, "reasons": ["no parsable segments"],
        }, []

    span = max(i["end"] for i in items)
    covered = sum(i["dur"] for i in items)
    idle_sec = sum(i["dur"] for i in items if i["label"] in IDLE_LABELS)

    by_label = defaultdict(list)
    for i in items:
        by_label[i["label"]].append(i["dur"])
    medians = {
        lab: st.median(ds)
        for lab, ds in by_label.items()
        if len(ds) >= MIN_REPEATS and lab not in IDLE_LABELS and st.median(ds) > 0
    }

    n_outlier = 0
    worst = 0.0
    excess = 0.0
    n_struggle = 0
    for i in items:
        med = medians.get(i["label"])
        i["ratio"] = (i["dur"] / med) if med else None
        i["is_outlier"] = bool(
            med and i["dur"] / med >= OUTLIER_RATIO and i["dur"] >= OUTLIER_MIN_SEC
        )
        i["is_idle"] = i["label"] in IDLE_LABELS
        i["is_struggle"] = bool(STRUGGLE_RE.search(i["label"]))
        if i["is_outlier"]:
            n_outlier += 1
            worst = max(worst, i["ratio"])
            excess += i["dur"] - med
        if i["is_struggle"]:
            n_struggle += 1

    idle_frac = idle_sec / span if span > 0 else 0.0
    coverage = covered / span if span > 0 else 0.0

    reasons = []
    if idle_frac >= 0.99:
        reasons.append("episode is entirely unlabelled (all segments null)")
    elif idle_frac > 0.10:
        reasons.append(f"{idle_frac:.0%} of runtime is idle/unlabelled")
    if n_outlier:
        reasons.append(
            f"{n_outlier} segment(s) run up to {worst:.1f}x the median for the "
            f"same action"
        )
    if n_struggle:
        reasons.append(f"{n_struggle} segment(s) use explicit recovery language")
    if coverage < 0.80:
        reasons.append(f"only {coverage:.0%} of runtime is annotated")

    feats = {
        "n_seg": n,
        "span": span,
        "coverage": coverage,
        "idle_frac": idle_frac,
        "idle_sec": idle_sec,
        "n_outlier": n_outlier,
        "worst_ratio": worst,
        "excess_sec": excess,
        "n_struggle": n_struggle,
        "lost_sec": idle_sec + excess,
        "flag": bool(reasons),
        "reasons": reasons,
    }
    return feats, items
