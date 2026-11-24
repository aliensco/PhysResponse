"""Reference-clip view of a result set: every g / v row of OWN is replaced by the reading of the same video under
its scene's reference clip for that parameter (rows of REF, items REF_ITEMS with key_own / clip_own), keeping the video's own key,
clip and prompted value (error and in-band recomputed against it); a video whose own clip is the reference keeps its row, and so
does one whose listed library clip has another camera or scene (event_ref.cross_view; ref_clip = ref_lib = own clip, ref_listed); other rows
unchanged. Every row of the view carries ref_clip (g / v) and ref_lib, the clip whose library gave the reading. Writes results_gen/<OUT>.jsonl and <OUT>_motion.jsonl in WORK.
Usage: python event_view.py WORK OWN REF REF_ITEMS_SET1[,SET2...] REFERENCE_CLIPS.json OUT"""
import glob, json, sys
from pathlib import Path
from event_ref import cross_view, library_clip   # F113 (C3): another clip's library only through the same view (PHYSPROBE_WORK = WORK)

W, OWN, REFN, SETS, RJ, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
R = W / 'results_gen'
REF = {tuple(k.split('|')): c for k, c in json.load(open(RJ)).items()}
READ = json.load(open(W / 'reference_reading.json')) if (W / 'reference_reading.json').exists() else None   # library clip of each reading (event_ref.py)
IT = {}
for s in SETS.split(','):
    for f in glob.glob(str(W / 'sam_gen' / s / 'items.json')): IT.update({x['key']: x for x in json.load(open(f))['items']})
alt, altm = {}, {}
def lib_of(key, p):   # the library clip the reading of the video should come from
    x = IT[key]
    return READ.get(f"{x['clip_own']}|{p}") if READ is not None else REF.get((x['clip'][:3], p))


for r in map(json.loads, open(R / f'{REFN}.jsonl')):
    if r.get('param') in ('g', 'v') and r['key'] in IT and lib_of(r['key'], r['param']) == r['clip']: alt[(IT[r['key']]['key_own'], r['param'])] = r
for o in map(json.loads, open(R / f'{REFN}_motion.jsonl')):
    if o.get('param') in ('g', 'v') and o['key'] in IT and lib_of(o['key'], o['param']) == o['clip']: altm[(IT[o['key']]['key_own'], o['param'])] = o
n = dict(replaced=0, own_is_reference=0, own_view=0, unchanged=0, missing=0, own_not_validated=0)
with open(R / f'{OUT}.jsonl', 'w') as o:
    for l in open(R / f'{OWN}.jsonl'):
        r = json.loads(l); p = r.get('param'); ref = REF.get((r.get('clip', '')[:3], p)) if p in ('g', 'v') else None
        if ref is None: n['unchanged'] += 1; o.write(l); continue
        if ref == r['clip']: n['own_is_reference'] += 1; o.write(json.dumps(dict(r, ref_clip=ref)) + '\n'); continue
        if cross_view(r['clip'], p, READ, REF):   # F113 (C3): the listed library clip has another camera or scene: the own-clip reading stays
            n['own_view'] += 1; o.write(json.dumps(dict(r, ref_clip=r['clip'], ref_lib=r['clip'], ref_listed=library_clip(r['clip'], p, READ, REF))) + '\n'); continue
        a = alt.get((r['key'], p))
        if a is None: n['missing'] += 1; continue
        x = dict(a, key=r['key'], clip=r['clip'], truth=r['truth'], ref_clip=ref, ref_lib=a['clip'], ref_truth=a.get('truth'), model=r['model'],
                 **{k: r[k] for k in ('seed', 'annotation', 'prompts') if k in r})
        t = r['truth']
        for k, e in (('est', 'err_pct'), ('best', 'best_err_pct')):
            if x.get(k) is not None and t: x[e] = 100 * (x[k] - t) / t
            else: x.pop(e, None)
        if x.get('band') and t:
            lo, hi = x['band']; lo = 0.0 if x['kind'] == 'upper_bound' else lo; hi = float('inf') if x['kind'] == 'lower_bound' else hi
            x['truth_in_band'] = bool(lo <= t <= hi)
        # F110 (2026-11-10, gate audit 4): the reading needs both gates, as mu (mu_ref.py): a video whose own-clip result fails the
        # known-truth gate (the instrument fails on the true physics of its own task and format) is not read through the reference clip
        if r.get('kind') == 'not_validated' and x.get('kind') != 'not_validated':
            x.update(kind_before_gate=x.get('kind'), kind='not_validated', not_validated_by=['own:' + w for w in r.get('not_validated_by') or []]); n['own_not_validated'] += 1
        o.write(json.dumps(x) + '\n'); n['replaced'] += 1
with open(R / f'{OUT}_motion.jsonl', 'w') as o:
    for l in open(R / f'{OWN}_motion.jsonl'):
        m = json.loads(l); a = altm.get((m['key'], m.get('param')))
        o.write(json.dumps(dict(a, key=m['key'], clip=m['clip'])) + '\n' if a else l)
print(OUT, n)
