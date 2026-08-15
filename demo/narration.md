# FailureRadar — demo narration

Six segments, ~2:45 total at Daniel's pace (~2.1 words/sec). Each renders to its own
MP3 so you can line each one up against a screen action without timing drift carrying
forward.

Record the screen silently first, then lay these under it. Where a segment says
`[on screen]`, that's your cue — it isn't spoken.

---

## 01_hook — ~25s
`[on screen] Dashboard loading, KPI row visible`

EgoVerse has four hundred forty-six thousand human demonstrations. Some teach a robot
the wrong thing, and nobody knows which.

There are no ground-truth labels. The success column reads True for four hundred
twenty-eight thousand episodes, and False for six. It's inert.

So we infer quality from the shape of the annotation timeline, and the geometry of the
hands.

---

## 02_meter — ~30s
`[on screen] Confidence meter tab, peeling_potatoes clip, click the red span`

A peeling task. Every peeling span here runs two to five seconds. Except this one, at
sixty.

That's the signal. For any label repeated three or more times, take its median
duration inside that episode, and flag spans running three times longer.

Normalising inside the episode is what matters: the more repetitive the task, the
tighter the baseline.

---

## 03_unlabelled — ~25s
`[on screen] Switch to the peeling_garlic clip`

A different failure. Nine spans, one hundred seconds, and every label is null. Someone
ran the segmenter and never wrote the text.

You'd assume that's idle footage. It isn't. Sixty percent of null spans move faster
than the twenty-fifth percentile of labelled work.

That's real manipulation with no supervision attached. Not data to discard — data to
recover.

---

## 04_validation — ~45s
`[on screen] Validation tab, scroll to the duration-bin table`

Those flags come from text. Do they mean anything physical?

We pulled end-effector pose for eight hundred episodes across sixty Modal containers —
one and a half percent of each store — and measured motion the detector never saw.

Flagged spans show far lower path efficiency. The hand moves, and arrives nowhere.

But efficiency falls with duration anyway, and flagged spans are long by definition.
So we compared each against normal spans of similar length. The gap holds in all seven
bins: one point five four times less efficient than duration alone predicts, at p
below ten to the minus thirteen.

A flag built from text predicts measured motion.

---

## 05_audit — ~22s
`[on screen] Corpus contract audit tab`

We also checked all four hundred forty-six thousand episodes against EgoVerse's own
rules.

Three hundred fifty-five thousand have no operator attribution. Forty-three thousand
store operators unhashed — including the raw names "ABC" and "Baoyu", which their
guide forbids. One contributor uses twenty-three thousand distinct task names.

---

## 06_close — ~18s
`[on screen] Tagged episodes tab, click Export drop-list`

Seven hundred fifty-three episodes flagged. Eight point six hours that teach a policy
nothing.

And it exports as a filter that pastes straight into EgoVerse's own download script.
Not a score to interpret — a diff to apply.

That's FailureRadar.
