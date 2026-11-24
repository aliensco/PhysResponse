"""Known-truth validation gate (issue log D28). A generated video's (task, parameter) is scored only where the evaluation
explains the TRUE physics of that task in the same video format: the reference known-answer result (results_reference/ref_*.json)
and the synthetic generated-like set of that format must both be an estimate (point / interval / bound / unidentifiable),
not not_explained / no_observation / error. Elsewhere the instrument itself fails on correct physics (tracking of stacked
or occluded objects, object rules with two objects), so a 'not explained' there says nothing about the model: the
generated result becomes kind 'not_validated' (the original kind is kept in kind_before_gate).
Format -> synthetic set: <= 24 frames -> s10f16 (DynamiCrafter); <= 16 fps -> s15v (16 fps x 81, PhysRVG 15 fps x 49);
otherwise s30f121 (24-25 fps x 121-189).
Usage: python validation_gate.py NAME [ITEMS_SET[,ITEMS_SET...]] [TRUTH_TAG] -> results_gen/<NAME>_gated.jsonl + results_gen/<NAME>_gate.json
(TRUTH_TAG: use the known-truth results of a rule variant, results_reference_<TAG>/ and results_gen/<set>_<TAG>.jsonl)"""
import collections, glob, json, sys

from paths import WORK as B
from order_rule import reading

PASS = {'point', 'interval', 'upper_bound', 'lower_bound', 'unidentifiable'}
SETS = ('s15v', 's30f121', 's10f16')
# F110 (2026-11-10, gate audit): (1) a known-truth result passes only when it finds the truth: the truth inside its band (one-sided for
# a bound), or its reading (order_rule.reading) within TRUTH_TOL of the truth, or, without a reading, the band within TRUTH_TOL of it;
# before, the kind alone passed (s15v_f0 T18_C01 v read 3.36 for 3.0, s30f121_f0 / s15v_f0 T20_C01 mu 0.57 for 1.0).
# (3) a (clip, parameter) without a row in its frame-0 format set (the reference-reading libraries T02_C02F, T16_C02F, T20_C02F) passed by
# lookup failure; it is now gated by the frame-4 set of the same format, and fails when that has no row either.
TRUTH_TOL = 0.10
# PHYSPROBE_GATE_TRUTH: where (1) applies - 'sets' (default: the format sets; the reference known-answer test by its kind as before:
# on the frame-4 reference videos all five T19 clips miss the truth by 13-75 % while their frame-0 format sets, from the package-reference
# retracks, all find it), 'all' (the reference test too), 'off' (old rule)
GATE_TRUTH = __import__('os').environ.get('PHYSPROBE_GATE_TRUTH', 'sets'); assert GATE_TRUTH in ('all', 'sets', 'off'), GATE_TRUTH
# PHYSPROBE_GATE_SHORT=NAME:SECONDS: videos lasting at most SECONDS are gated by their own known-truth set NAME (e.g. s2s:2.5 for
# 2 s clips, made with make_synth.py --seconds 2); default unset (the official gate)
_SH = __import__('os').environ.get('PHYSPROBE_GATE_SHORT', '')
SHORT = (_SH.split(':')[0], float(_SH.split(':')[1])) if _SH else None
if SHORT: SETS = SETS + (SHORT[0],)


def format_set(fps, frames):
    if SHORT and fps and (frames - 1) / fps <= SHORT[1]: return SHORT[0]
    return 's10f16' if frames <= 24 else ('s15v' if fps <= 16 else 's30f121')


def contains_truth(r):
    """F110 (1): the truth inside the band (one-sided for a bound); else the reading within TRUTH_TOL of the truth, or, without a
    reading, the band within TRUTH_TOL of it"""
    if r.get('truth_in_band'): return True
    b, t = r.get('band'), r.get('truth')
    if not b or not t: return False
    v = reading(r)
    if v is not None: return abs(v / t - 1) <= TRUTH_TOL
    lo, hi = (0.0 if r['kind'] == 'upper_bound' else b[0]), (float('inf') if r['kind'] == 'lower_bound' else b[1])
    return (lo / t - 1 if t < lo else 1 - hi / t if t > hi else 0.0) <= TRUTH_TOL


def failed(tag='', have=None):
    """(set name) -> {(clip, param)} where the known-truth result is not an estimate or misses the truth (F110); 'reference' =
    results_reference. have (dict, filled when given): (set name) -> {(clip, param)} with a known-truth result."""
    bad = collections.defaultdict(set); have = collections.defaultdict(set) if have is None else have
    for f in glob.glob(str(B / ('results_reference' + tag) / 'ref_*.json')):
        if f.endswith('_summary.json'): continue
        r = json.load(open(f)); have['reference'].add((r['clip'], r['param']))
        if r.get('kind') not in PASS or (GATE_TRUTH == 'all' and not contains_truth(r)): bad['reference'].add((r['clip'], r['param']))
    for s in SETS + tuple(f'{x}_f0' for x in SETS if (B / 'results_gen' / f'{x}_f0{tag}.jsonl').exists()):   # frame-0 sets, see F0 below
        for line in open(B / 'results_gen' / f'{s}{tag}.jsonl'):
            r = json.loads(line)
            if 'param' not in r: continue
            have[s].add((r['clip'], r['param']))
            if r.get('kind') not in PASS or (GATE_TRUTH != 'off' and not contains_truth(r)): bad[s].add((r['clip'], r['param']))
    return bad


if __name__ == '__main__':
    name = sys.argv[1]; items_set = sys.argv[2] if len(sys.argv) > 2 else 'all'; tag = '_' + sys.argv[3] if len(sys.argv) > 3 else ''
    fmt = {x['key']: format_set(x['fps'], x['frames']) + ('_f0' if x.get('condition_frame_index', 4) == 0 else '')   # F0: videos generated
           for n in items_set.split(',')                                      # from frame 0 are gated by the frame-0 known-truth set of their format
           for x in json.load(open(B / 'sam_gen' / n / 'items.json'))['items']}   # (s30f121_f0 / s15v_f0, 2026-11-04); several item sets: the later ones win
    missing = {s for s in fmt.values() if s.endswith('_f0') and not (B / 'results_gen' / f'{s}{tag}.jsonl').exists()}
    if missing: sys.exit(f'frame-0 known-truth sets missing: {sorted(missing)}')
    have = collections.defaultdict(set); bad = failed(tag, have); n = collections.Counter(); out = []
    for line in open(B / 'results_gen' / f'{name}.jsonl'):
        r = json.loads(line)
        if 'param' in r:
            s = fmt[r['key']]; cp = (r['clip'], r['param'])
            if cp not in have[s] and s.endswith('_f0') and cp in have[s[:-3]]: s = s[:-3]   # F110 (3): the frame-4 set of the format
            why = [k for k in ('reference', s) if cp in bad[k]] + ([f'{s}:no_known_truth'] if cp not in have[s] else [])
            r['format_set'] = s
            if why:
                r.update(kind_before_gate=r.get('kind'), kind='not_validated', not_validated_by=why); n[(s, r['param'], 'gated')] += 1
            n[(s, r['param'], 'all')] += 1
        out.append(r)
    with open(B / 'results_gen' / f'{name}_gated.jsonl', 'w') as f:
        for r in out: f.write(json.dumps(r) + '\n')
    rep = {'failed_known_truth': {k: sorted(f'{c}|{p}' for c, p in v) for k, v in sorted(bad.items())},
           'gated_share': {f'{s}|{p}': f"{n[(s, p, 'gated')]}/{n[(s, p, 'all')]}" for s in SETS + tuple(f'{x}_f0' for x in SETS) for p in ('g', 'mu', 'v') if n[(s, p, 'all')]}}
    json.dump(rep, open(B / 'results_gen' / f'{name}_gate.json', 'w'), indent=1)
    for k, v in rep['failed_known_truth'].items(): print(k, len(v), ' '.join(v))
    print(json.dumps(rep['gated_share']))
