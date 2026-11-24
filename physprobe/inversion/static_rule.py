"""Static-reference clips (rule D-S1): the reference video of these clips does not move, so the simulator has nothing to fit. A
generated video is scored by whether it stays still: largest change of any box edge from frame 0 (3-frame median filter), whole
video, every tracked object, in camera pixels; still if <= tau = max(3 px, 2 x the same measure on the reference video), else
moved. Rows (one per video and parameter g, mu): kind static_still (explained) / static_moved (not explained) / no_observation,
truth = the prompted value, no best value. Clips listed in not_scored keep no result of their own.
Inputs in the base: static_rule.json (clips, not_scored, truths) and static_reference_tracks.jsonl (tracks of the reference videos).
Usage (from physprobe/inversion, PHYSPROBE_WORK set): python static_rule.py NAME
  -> results_gen/NAME.jsonl and NAME_motion.jsonl with the rows of these clips replaced, results_gen/NAME_static.json"""
import json, sys
import numpy as np

from paths import WORK as B

N = sys.argv[1]; R = B / 'results_gen'
S = json.load(open(B / 'static_rule.json')); CLIPS = set(S['clips']); SKIP = set(S['not_scored']); STATIC_PX = 3.0
TRUTH = {(c, p): v for c, d in S['truths'].items() for p, v in d.items()}


def med3(x):
    y = x.copy()
    for i in range(1, len(x) - 1): y[i] = np.nanmedian(x[i - 1:i + 2], axis=0)
    return y


def edge_change(t, scale=(1.0, 1.0)):
    x = np.array([b if b else [np.nan] * 4 for b in t['xyxy']], float)
    if not np.isfinite(x[0]).all() or len(x) < 2: return float('nan')
    x = med3(x); d = np.abs(x - x[0]); sx, sy = scale
    d[:, [0, 2]] /= sx; d[:, [1, 3]] /= sy
    return float(np.nanmax(d)) if np.isfinite(d).any() else float('nan')


def own_clip(r): return r['key'].split('.')[1] if 'key' in r else r.get('clip')   # the video's own clip (view rows carry the reference clip in 'clip')


noise = {}
for l in open(B / 'static_reference_tracks.jsonl'):
    r = json.loads(l); c, o = r['id'].split('__')
    if c in CLIPS: noise[c] = max(noise.get(c, 0.0), edge_change(r['tracks'][o]))
tau = {c: max(STATIC_PX, 2 * noise[c]) for c in CLIPS}
items = {x['key']: x for x in json.load(open(B / 'sam_gen' / N / 'items.json'))['items'] if x['clip'] in CLIPS}
mv = {}
for l in open(B / 'sam_gen' / N / 'tracks_camcomp.jsonl'):
    r = json.loads(l); k, o = r['id'].split('__', 1)
    if k not in items or r.get('exception') or o not in (r.get('tracks') or {}): continue
    sc = (items[k].get('cam_to_video') or {}).get('scale') or [1.0, 1.0]
    mv.setdefault(k, {})[o] = edge_change(r['tracks'][o], sc)
rows, rep = [], dict(rule='D-S1', noise_px=noise, tau_px=tau, not_scored=sorted(SKIP), videos={})
for k, x in sorted(items.items()):
    m = mv.get(k, {}); v = [d for d in m.values() if d == d]; mx = max(v) if v else None; c = x['clip']
    kind = 'no_observation' if mx is None else ('static_still' if mx <= tau[c] else 'static_moved')
    rep['videos'][k] = dict(max_edge_px=None if mx is None else round(mx, 2), kind=kind, scored=c not in SKIP)
    for p in ('g', 'mu') if c not in SKIP else ():
        rows.append(dict(clip=c, param=p, truth=TRUTH[(c, p)], kind=kind, est=None, best=None, band=None, truth_in_band=None, static_rule='D-S1',
                         max_edge_px=rep['videos'][k]['max_edge_px'], tau_px=tau[c], key=k, model=x['model'], seed=x.get('seed'), annotation=x.get('annotation')))
for name, add in ((f'{N}.jsonl', rows), (f'{N}_motion.jsonl', [])):
    keep = [l for l in open(R / name) if l.strip() and own_clip(json.loads(l)) not in CLIPS]
    open(R / name, 'w').write(''.join(keep) + ''.join(json.dumps(r) + '\n' for r in add))
json.dump(rep, open(R / f'{N}_static.json', 'w'), indent=1)
print(len(items), 'videos of the static clips,', len(rows), 'rows', {kd: sum(v['kind'] == kd for v in rep['videos'].values()) for kd in ('static_still', 'static_moved', 'no_observation')})
