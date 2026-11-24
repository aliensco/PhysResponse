"""D69 / D71 (2026-11-10): the reading of a result and the measured order of two results for the concordance measures. A point or interval result orders by its best
value; a one-sided bound orders by the range it allows (upper bound: (0, band high], lower bound: [band low, inf)), so it orders against
another result only when the two ranges do not overlap. The best value of a bound inside a flat side of the loss is not a reading
(physically correct T01_C03: g <= 16.5 with best 0.63 under the old shift rule), so it no longer decides an order."""
import math
import numpy as np


RATIO = 3.0   # D71: a reading needs a two-sided band no wider than this factor (high / low)


def reading(r):
    """D71 (2026-11-10: one value per result, small errors): the reading of a result is its best value when the evaluator pins
    the parameter down - a point, or an interval whose band spans at most a factor RATIO. One-sided bounds, wider intervals,
    unidentifiable, not explained and unobserved results have no reading (on the physically correct renders the best value of a bound
    can sit far from the truth on the flat side of the loss: T13_C03 g 0.63 for 9.81, T17_C05 mu 0.745 for 0.28; with RATIO 3 the
    largest error of a reading is 20 % for g, 17 % for v and 49 % for mu, the last from a mu grid that misses the truth, T20_C01)."""
    k, b, v = r.get('kind'), r.get('band'), r.get('best')
    if not v: return None
    if k == 'point': return float(v)
    if k == 'interval' and b and b[0] > 0 and b[1] / b[0] <= RATIO: return float(v)
    return None


def span(r):
    """the range a result allows for the order: its reading, else its band (a bound open on one side)"""
    v = reading(r)
    if v is not None: return v, v
    b = r.get('band')
    if r['kind'] == 'upper_bound': return 0.0, float(b[1] if b else r['best'])
    if r['kind'] == 'lower_bound': return float(b[0] if b else r['best']), float('inf')
    if b: return float(b[0]), float(b[1])
    return float(r['best']), float(r['best'])


def order_sign(x, y):
    """+1 if x reads above y, -1 below, 0 if the order is not measured"""
    (lx, hx), (ly, hy) = span(x), span(y)
    return 1 if lx > hy else -1 if hx < ly else 0


def order_matrix(lo, hi):
    """pairwise signs for arrays of ranges: +1 where row i lies above column j, -1 below, 0 overlapping"""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return (lo[:, None] > hi[None, :]).astype(int) - (hi[:, None] < lo[None, :]).astype(int)


TIER = {'interval': 1, 'upper_bound': 2, 'lower_bound': 2, 'unidentifiable': 3, 'not_explained': 4, 'not_validated': 4}
COVERAGE = 0.10   # main-table error: the most confident 10 % of all results of a model (2026-11-10)


def ranked_errors(rows):
    """Risk-coverage (2026-11-10: models differ in how many results the evaluator can measure, so errors over each
    model's own readings compare different samples). Every result with a value is ranked by how firmly the evaluator measured it:
    readings first, then wider intervals, one-sided bounds, unidentifiable and not explained results (value = the best fit); within a
    tier the smaller best-fit box-edge residual first. Returns the relative errors |value / prompted - 1| in that order (results without
    a value - no observation, still - are left out but stay in the denominator of the coverage)."""
    out = []
    for r in rows:
        v = reading(r); t = 0
        if v is None: t = TIER.get(r.get('kind')); v = r.get('best')
        if t is None or not v or not r.get('truth'): continue
        out.append(((t, r.get('best_rms_px_max') or 0.0), abs(v / r['truth'] - 1)))
    return [e for _, e in sorted(out, key=lambda x: x[0])]


def risk(errors, n_all, c=COVERAGE):
    """median error of the most confident share c of all n_all results; None if fewer results have a value than that share"""
    k = int(math.floor(c * n_all + 0.5 + 1e-9))   # F118 (2026-11-11): half up, robust to float grids
    if k < 1 or k > len(errors): return None
    return float(np.median(errors[:k]))


def risk_curve(errors, n_all, step=0.01):
    """(coverage, risk) for every coverage step the results reach"""
    return [(round(c, 2), risk(errors, n_all, c)) for c in np.arange(step, 1 + 1e-9, step) if risk(errors, n_all, c) is not None]
