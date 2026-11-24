"""Reference-clip reading of one batch: every video is also inverted against its scene's reference clip for g and for v
(REFERENCE_CLIPS.json: {"<scene>|<param>": "<clip>"}; in the evaluation base reference_reading.json gives per clip the library of
its own sample to read with, or none), reusing its camera-compensated tracks, so that the videos of one scene with
different prompted values are measured with the same library, window and gate. A video whose own clip is the reference needs no
second inversion. Another clip's library is used only when the two clips share the view (F113): the reading projects that
clip's simulations with that clip's camera but maps them into the video with the video's own registration, so a different
camera or scene makes the reading meaningless (T03 g, T18 g and mu: other camera; T06_C04: other scene); such a video keeps
its own-clip reading (event_view.py, mu_ref.py). Writes sam_gen/<BATCH>_ref/items.json and tracks_camcomp.jsonl (keys
<key>.ref.<clip>, with clip_own, key_own, ref_params); exit 3 when nothing needs a second inversion. Then run
eval_generated.py <BATCH>_ref ... --params g,v.
Usage (from physprobe/inversion, PHYSPROBE_WORK set): python event_ref.py BATCH REFERENCE_CLIPS.json"""
import collections, functools, json, os, sys
from pathlib import Path
sys.path.insert(0, os.getcwd())
import numpy as np
import common as c2

VIEW_PX = 1.0   # two clips share the camera when their cameras put the scene objects within this many pixels of each other
SAME_IMAGE = 0.5   # ... and their condition images agree (cam_comp agreement) at least this much without registration, and
SAME_GAIN = 0.05   # ... registering them does not raise the agreement by more than this


def _project(cam, X):
    eye, fwd, right, up, F, Wd, Ht = cam; P = np.asarray(X, float) - eye; z = P @ fwd
    return np.stack([Wd / 2 + F * (P @ right) / z, Ht / 2 - F * (P @ up) / z], 1)


@functools.lru_cache(maxsize=None)
def _camera(c): return c2.camera(c)


@functools.lru_cache(maxsize=None)
def _positions(c):
    try: return tuple(tuple(q[:3]) for q in c2.initial_poses(c).values())
    except Exception: return ()   # no recorded or replayed poses (T06_C04): the other clip's, else a fixed grid


@functools.lru_cache(maxsize=None)
def view_gap(a, b):
    """largest pixel distance between the projections of the two clips' objects (condition-frame positions) through the
    benchmark cameras of clips a and b; inf for different image sizes"""
    ca, cb = _camera(a), _camera(b)
    if tuple(ca[5:]) != tuple(cb[5:]): return float('inf')
    X = list(_positions(a) + _positions(b))
    if not X: X = [ca[0] + ca[1] * d + ca[2] * x + ca[3] * y for d in (1.0, 2.0) for x in (-0.5, 0.5) for y in (-0.3, 0.3)]
    return float(np.hypot(*(_project(ca, X) - _project(cb, X)).T).max())


@functools.lru_cache(maxsize=None)
def same_image(a, b):
    """the condition images of clips a and b (evaluation base conditions/<clip>.png) show the same view: they agree without
    registration, and an affine registration does not explain them clearly better; True when an image is missing (the
    formal-library variants <clip>F have none: the camera check alone)"""
    import cv2, cam_comp as cc
    C = c2.B / 'conditions'
    if not C.is_dir() and (c2.B / 'WORKSPACE_OK').exists(): C = Path(open(c2.B / 'WORKSPACE_OK').read().strip()) / 'conditions'   # a run workspace made before the base had conditions/
    pa, pb = C / f'{a}.png', C / f'{b}.png'
    if not (pa.exists() and pb.exists()): return True
    g = [cv2.cvtColor(cv2.resize(cv2.imread(str(p)), cc.SZ, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 for p in (pa, pb)]
    d = cc.bandpass(g[0]); tex = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), cc.BAND)) > 0.01
    s0 = cc.agreement(d, cc.warped(g[1], cc.I3), tex); M = cc.ecc(g[0], g[1], cc.I3, None)
    s1 = cc.agreement(d, cc.warped(g[1], M), tex) if M is not None else 0.0
    return s0 >= SAME_IMAGE and s0 >= s1 - SAME_GAIN


def same_view(a, b):
    return a == b or (view_gap(a, b) <= VIEW_PX and same_image(a, b))


def library_clip(clip, p, READ, REF):
    """clip whose library the reading as listed uses (reference_reading.json, else the scene's reference clip); may be None"""
    return READ.get(f'{clip}|{p}') if READ is not None else REF.get((clip[:3], p))


def cross_view(clip, p, READ, REF):
    """the listed library clip of (clip, p) is another clip that does not share the view: the reading stays on the own clip"""
    rc = library_clip(clip, p, READ, REF)
    return bool(rc) and rc != clip and not same_view(clip, rc)


def reading_clip(clip, p, READ, REF):
    """clip whose library gives the reading of a video of clip for parameter p (the clip itself when the listed one has another view)"""
    return clip if cross_view(clip, p, READ, REF) else library_clip(clip, p, READ, REF)


def load_reading():
    RR = c2.B / 'reference_reading.json'   # evaluation base: per clip and parameter the library clip of the reading (null: none of its sample)
    return json.load(open(RR)) if RR.exists() else None


def main():
    N, RJ = sys.argv[1:3]
    REF = {tuple(k.split('|')): c for k, c in json.load(open(RJ)).items()}
    READ = load_reading()
    src = c2.B / 'sam_gen' / N; dst = c2.B / 'sam_gen' / f'{N}_ref'
    d = json.load(open(src / 'items.json'))
    sel = []; own = collections.Counter()
    for x in d['items']:
        if not x.get('map'): continue
        refs = collections.defaultdict(list)
        for p in ('g', 'v'):
            if cross_view(x['clip'], p, READ, REF): own[f"{x['clip']}|{p}"] += 1; continue
            rc = library_clip(x['clip'], p, READ, REF)
            if rc and rc != x['clip']: refs[rc].append(p)
        for rc, ps in sorted(refs.items()):
            sel.append(dict(x, key=f"{x['key']}.ref.{rc}", clip=rc, clip_own=x['clip'], key_own=x['key'], ref_params=ps))
    if own: print('own-clip reading (the listed library clip has another view):', dict(sorted(own.items())))
    if not sel: print('no reference-clip inversion needed'); sys.exit(3)
    dst.mkdir(parents=True, exist_ok=True)
    json.dump(dict(items=sel, model_median_map=d.get('model_median_map')), open(dst / 'items.json', 'w'), indent=1)
    by = collections.defaultdict(list)
    for x in sel: by[x['key_own']].append(x['key'])
    n = 0
    with open(dst / 'tracks_camcomp.jsonl', 'w') as o:
        for l in open(src / 'tracks_camcomp.jsonl'):
            k, _, obj = json.loads(l)['id'].partition('__')
            for nk in by.get(k, ()):
                r = json.loads(l); r['id'] = f'{nk}__{obj}'; o.write(json.dumps(r) + '\n'); n += 1
    print(len(sel), 'reference-clip inversions,', n, 'track rows')


if __name__ == '__main__':
    main()
