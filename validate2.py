"""Duration-controlled validation.

Efficiency falls with duration by construction, and flagged segments are long
by definition. So compare each outlier only against NON-outlier segments of
comparable duration, and against the fitted duration->efficiency trend.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

seg = pd.read_parquet("kinematics_segments.parquet")
seg["label_n"] = seg["label"].fillna("").str.strip().str.lower()
IDLE = {"none", "no action", "", "idle", "nothing"}

med = (seg[~seg["label_n"].isin(IDLE)]
       .groupby(["episode_hash", "label_n"])["dur"]
       .agg(["median", "count"]).rename(columns={"median": "med_dur", "count": "n_rep"}))
seg = seg.merge(med, on=["episode_hash", "label_n"], how="left")
seg["ratio"] = seg["dur"] / seg["med_dur"]
seg["is_outlier"] = (seg["n_rep"] >= 3) & (seg["ratio"] >= 3.0) & (seg["dur"] >= 3.0)
seg["is_idle"] = seg["label_n"].isin(IDLE)

work = seg[~seg["is_idle"]].copy()

print("=" * 78)
print("DURATION-MATCHED: efficiency of outliers vs non-outliers, same duration bin")
print("=" * 78)
bins = [3, 5, 8, 12, 20, 35, 60, 1000]
work["bin"] = pd.cut(work["dur"], bins)
rows = []
for b, g in work.groupby("bin", observed=True):
    o = g[g["is_outlier"]]
    c = g[~g["is_outlier"]]
    if len(o) >= 5 and len(c) >= 5:
        rows.append({
            "duration_bin": str(b),
            "n_outlier": len(o),
            "n_other": len(c),
            "eff_outlier": o["efficiency"].median(),
            "eff_other": c["efficiency"].median(),
            "ratio": o["efficiency"].median() / c["efficiency"].median(),
            "dwell_outlier": o["dwell_frac"].median(),
            "dwell_other": c["dwell_frac"].median(),
        })
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 78)
print("RESIDUAL TEST: efficiency vs log(duration) trend fitted on NON-outliers")
print("=" * 78)
c = work[~work["is_outlier"]]
x = np.log(c["dur"].values)
y = np.log(c["efficiency"].values.clip(1e-4))
A = np.vstack([x, np.ones_like(x)]).T
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
print(f"fit: log(eff) = {coef[0]:.3f}*log(dur) + {coef[1]:.3f}")


def resid(d):
    xx = np.log(d["dur"].values)
    yy = np.log(d["efficiency"].values.clip(1e-4))
    return yy - (coef[0] * xx + coef[1])


r_out = resid(work[work["is_outlier"]])
r_ctl = resid(c)
print(f"\nresidual (log-eff vs trend):")
print(f"  outliers      n={len(r_out):5d}  mean={r_out.mean():+.3f}  median={np.median(r_out):+.3f}")
print(f"  non-outliers  n={len(r_ctl):5d}  mean={r_ctl.mean():+.3f}  median={np.median(r_ctl):+.3f}")
print(f"  => outliers are {np.exp(np.median(r_ctl) - np.median(r_out)):.2f}x less "
      f"efficient than duration alone predicts")

def mannwhitney_p(a, b):
    """Two-sided Mann-Whitney U, normal approximation with tie correction."""
    from math import erfc, sqrt

    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    s = np.sort(allv)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            idx = order[i:j + 1]
            ranks[idx] = (i + j + 2) / 2.0
        i = j + 1
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sd = sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (u1 - mu) / sd
    return erfc(abs(z) / sqrt(2))


print(f"  Mann-Whitney p = {mannwhitney_p(r_out, r_ctl):.3g}")

print("\n" + "=" * 78)
print("UNLABELLED EPISODES: is this idle time, or lost data?")
print("=" * 78)
idle = seg[seg["is_idle"]]
lab = seg[~seg["is_idle"]]
print(f"null-labelled segments: {len(idle)}  ({idle['dur'].sum() / 3600:.2f} h)")
print(f"labelled segments:      {len(lab)}  ({lab['dur'].sum() / 3600:.2f} h)")
print(f"\nmedian hand speed  null={idle['mean_speed'].median():.5f}  "
      f"labelled={lab['mean_speed'].median():.5f}")
print(f"median dwell_frac  null={idle['dwell_frac'].median():.3f}  "
      f"labelled={lab['dwell_frac'].median():.3f}")
thr = lab["mean_speed"].quantile(0.25)
active = idle[idle["mean_speed"] > thr]
print(f"\nnull-labelled segments moving faster than the 25th pct of LABELLED work: "
      f"{len(active)}/{len(idle)} ({100 * len(active) / len(idle):.0f}%)")
print(f"  => {active['dur'].sum() / 3600:.2f} h of active manipulation carries no label")
