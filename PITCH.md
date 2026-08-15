# FailureRadar — demo notes

**Live:** https://panforrest.github.io/failureradar/
**Repo:** https://github.com/panforrest/failureradar

---

## The 30-second version

> EgoVerse has no ground-truth success labels — `is_eval` is False for all 446,957
> episodes and `eval_success` is True for 428,310 of them and False for six. So we
> inferred demonstration quality from two things already in the data: the shape of the
> annotation timeline, and the geometry of the hands. We flagged 753 episodes holding
> 8.67 hours of runtime that teaches a policy nothing — and then proved the flags
> mean something by checking them against motion data the detector never saw.

---

## Three-minute demo path

**1. Open on the Confidence meter tab — lead with `peeling_potatoes`.**

Point at the strip under the video. "Peeling spans in this episode run 2.7 to 5.5
seconds each. This one runs 60.1 seconds — twelve times the median for the exact same
action, in the same episode." Click the red span; the video seeks there.

Say the important part out loud: *the red band was drawn from annotation timestamps.
The blue line under it is hand speed from the pose arrays. Nothing about the flag used
the motion data.*

**2. Switch clips to `peeling_garlic`.**

"Nine spans, one hundred seconds, and every single label is null. Someone ran the
segmenter and never wrote the text." Then the twist: "You'd assume that's idle
footage. It isn't — 60.6% of null spans move faster than the 25th percentile of
labelled work. That's 1.62 hours of real manipulation with no language supervision
attached. That's not data to drop, it's data to recover."

**3. Validation tab — this is the one that wins arguments.**

"The obvious objection is that path efficiency falls with duration anyway, and our
flagged spans are long by definition. So we compared each flagged span only against
non-flagged spans of comparable length. The gap survives in all seven duration bins.
Against a fitted trend, flagged spans are 1.54× less efficient than duration alone
predicts, p = 1.6 × 10⁻¹⁴."

**4. Corpus audit tab — the finding they didn't ask for.**

"While we were in there: 355,927 episodes have no operator attribution, 43,271 have an
operator field that isn't a hash — including the literal strings `ABC` and `Baoyu`,
which their own contributing guide says must never be stored raw. 386,059 have no
license, and one contributor is using 23,013 distinct task names."

**5. Tagged episodes tab — close on the action.**

Hit **Export drop-list**. "This is a `DatasetFilter` that pastes straight into
EgoVerse's own `sync_s3.py`. You don't get a score, you get a diff."

---

## Numbers worth memorising

| | |
|---|---|
| Corpus | 446,957 episodes, 4,002.7 h |
| Audited | 23,289 (every episode carrying annotations) |
| Flagged | 753 — 3.23% |
| Wasted runtime | 8.67 h |
| Kinematic scan | 800 episodes, 7,518 spans, 60 Modal containers, ~3 min, 0 failures |
| Bytes fetched per episode | 4.4 MB of 271 MB — **1.6%** |
| Efficiency penalty | 1.54× below the duration trend, p = 1.6×10⁻¹⁴ |
| Duration bins where the gap holds | 7 of 7 |
| Unlabelled spans that are actually active | 60.6%, 1.62 h |

---

## Questions you should expect

**"3.23% seems low — is this worth doing?"**
The base rate *is* the finding; most mecka demonstrations are clean, which is worth
knowing. But note what 3.23% buys: 8.67 hours of runtime removed from a 23k-episode
slice, found for about a dollar of compute, with no human review and no LLM judge. And
the flag rate isn't uniform — the per-task table shows where it concentrates, which is
the actual sampling guidance.

**"How do you know these are failures? You have no ground truth."**
We don't claim they're failures, and that's deliberate. Nobody can claim that —
EgoVerse ships no usable success labels. We claim something narrower and checkable:
this runtime teaches a policy nothing, because the hand is moving without making
progress or isn't labelled at all. That's a defensible claim about training value, not
a guess about human intent.

**"Isn't low path efficiency just what long segments look like?"**
That was our first worry too, so we controlled for it two ways — duration-matched bins
and a residual test against a fitted trend. Both hold. Details on the Validation tab.

**"Could these be annotation errors rather than genuine struggles?"**
Sometimes, and it doesn't change the conclusion — a span mislabelled as one action
for 60 seconds is defective training data either way. But the kinematics say the hand
really is inefficient in those windows, so it isn't purely a labelling artifact.

**"Why only mecka? That's 5% of the corpus."**
Because mecka is the only contributor that publishes segment annotations. That's a
finding, not a limitation we're hiding: 95% of EgoVerse is currently unauditable for
demonstration quality, and the fix is a contribution-format requirement.

**"How does this scale to all 446,957 episodes?"**
The annotation path is free — it's a SQL column, it already ran over everything
annotated. The kinematic path needs no annotations at all: cycle structure can come
from autocorrelation of the pose signal instead of from labels. At 4.4 MB and ~11 GETs
per episode, the full corpus is roughly 2 TB of transfer — a few hours on a few
hundred Modal containers, well inside a normal preprocessing budget.

**"What did you get wrong?"**
Our first detector counted repeated annotation labels, assuming humans repeat actions
when they fail. Its top hit was someone scooping coffee beans 66 times — a perfect
demonstration. Repetition is a property of the task, not of failure. We kept that
script in the repo. The fix was normalising within the episode, which is what makes
the current detector work.

---

## Where Modal earned its place

- **Fan-out:** `scan_episode.map()` over 60 containers — 800 episodes in ~3 minutes,
  zero failures. Serially this is over an hour.
- **Volume:** `egoverse-cache` holds the 447k-row episode table, R2 credentials, scan
  results and clips, so nothing is recomputed between runs.
- **Secrets:** credentials injected at runtime, never baked into an image.
- **Selective fetch:** listing each Zarr store and pulling only the pose arrays is a
  60× transfer reduction — the reason this was possible on venue wifi at all.
- Windows note: `modal shell` is unsupported there, so everything runs through
  `modal run` entrypoints instead.
