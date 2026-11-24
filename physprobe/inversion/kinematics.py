"""Kinematics of every evaluated object, video against the simulation at the prompted values (same window and frame map as
motion_check.py: s = cf + (k - kc) DT / dt, predictability window, 3-frame median filter, camera-compensated tracks).
Per object of every validated g result: the box-centre displacement curves d(t) of the video and of the simulation (up to 48 points
each, for plots), their largest values, the exponent p of d = a (t - t0)^p on the rising phase (p = 2: uniform acceleration from
rest, p = 1: constant speed), the time to half the largest displacement (t50), the path deviation (median distance of the moving
video centre to the simulated centre path, over the path length) and the progress (arc position along the simulated path of the
video centre at its largest displacement, over the path length) and the displacement along the simulated direction r_dir
(2026-11-10): the video displacement at the end of the simulated window projected on the simulated displacement there, over its
length (1 = right amount and direction, 0 = no motion or sideways, < 0 = backwards; an object lost earlier keeps its last observed
position; simulated displacement >= 20 px).
Usage (PHYSPROBE_WORK = the run dir, from physprobe/inversion): python kinematics.py RESULTS ITEMS_SET TRACKS OUT.jsonl [clock] [workers]
(the one-command evaluation runs it as the 'kinematics' step; main_readme_all.py --kin turns the rows into the three kinematic columns)"""
import collections, json, math, os, sys
from multiprocessing import get_context
import numpy as np
sys.path.insert(0, os.environ.get('PHYSPROBE_CODE', os.getcwd()))
import evaluator as ev

NAME, ISET, TRACKS, OUT = sys.argv[1:5]; CLOCK = sys.argv[5] if len(sys.argv) > 5 else 'realtime'; WORKERS = int(sys.argv[6]) if len(sys.argv) > 6 else 16
items = {x['key']: x for x in json.load(open(ev.B / 'sam_gen' / ISET / 'items.json'))['items'] if x.get('map') or x.get('cam_to_video')}
R = [r for r in map(json.loads, open(ev.B / 'results_gen' / f'{NAME}.jsonl')) if r.get('param') == 'g' and r.get('key') in items]   # every g result, any kind (2026-11-10: still / gated videos enter the curves and r_dir)
OBS = collections.defaultdict(dict); KC = {}
for line in open(TRACKS):
    r = json.loads(line); vid, n = r['id'].split('__', 1)
    if vid not in items or n not in (r.get('tracks') or {}): continue
    tr = r['tracks'][n]
    OBS[vid][n] = (np.array(tr['frames']), np.array([b if b is not None else [np.nan] * 4 for b in tr['xyxy']], float)); KC[vid] = r.get('prompt_frame', 0)
by_clip = collections.defaultdict(list)
for r in R: by_clip[r['clip']].append(r)


def medfilt3(x):
    y = x.copy()
    for i in range(1, len(x) - 1): y[i] = np.nanmedian(x[i - 1:i + 2], axis=0)
    return y


def fit_p(t, d):
    ok = np.isfinite(d); t, d = t[ok], d[ok]
    if len(d) < 6 or d.max() < 10: return None
    dm = d.max(); im = int(np.argmax(d >= 0.95 * dm)); sel = (d > 0.05 * dm) & (np.arange(len(d)) <= im)
    if sel.sum() < 5: return None
    ts, ds = t[sel], d[sel]; step = np.median(np.diff(t)) if len(t) > 1 else 0.03; best = None
    for t0 in ts[0] - step * np.linspace(0.05, 12, 60):
        A = np.vstack([np.log(ts - t0), np.ones_like(ts)]).T; y = np.log(ds); coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = float(((A @ coef - y) ** 2).sum())
        if best is None or r < best[0]: best = (r, float(coef[0]))
    return round(best[1], 3)


def t_half(t, d):
    ok = np.isfinite(d)
    if ok.sum() < 3 or np.nanmax(d) < 10: return None
    return float(t[ok][np.argmax(d[ok] >= 0.5 * np.nanmax(d))])


def path_stats(cobs, csim):
    seg = np.diff(csim, axis=0); L = np.linalg.norm(seg, axis=1); tot = float(L.sum())
    if tot < 20 or len(cobs) < 3: return None, None
    cum = np.concatenate([[0], np.cumsum(L)])
    def proj(p):
        best = (1e18, 0.0)
        for i, (a, s, l) in enumerate(zip(csim[:-1], seg, L)):
            u = 0.0 if l == 0 else float(np.clip(np.dot(p - a, s) / (l * l), 0, 1)); q = a + u * s; dd = float(np.linalg.norm(p - q))
            if dd < best[0]: best = (dd, cum[i] + u * l)
        return best
    mv = np.linalg.norm(cobs - cobs[0], axis=1) > 3
    if mv.sum() < 2: return None, 0.0
    dev = float(np.median([proj(p)[0] for p in cobs[mv]])) / tot
    prog = proj(cobs[int(np.nanargmax(np.linalg.norm(cobs - cobs[0], axis=1)))])[1] / tot
    return round(dev, 4), round(prog, 4)


def thin(t, d, n=48):
    ok = np.isfinite(d); t, d = t[ok], d[ok]
    if len(t) > n: i = np.linspace(0, len(t) - 1, n).round().astype(int); t, d = t[i], d[i]
    return [[round(float(a), 4), round(float(b), 2)] for a, b in zip(t, d)]


def run_clip(clip):
    out = []
    try:
        dt, _ = ev.clock(clip); hz = ev.HZ[clip]['objects']; lib = ev.library(clip, 'g', workers=1)
    except Exception as e: return out
    if not lib: return out
    truth = ev.truth_of(clip, 'g'); _, boxes = min(lib, key=lambda t: abs(math.log(t[0] / truth)))
    for r in by_clip[clip]:
        x = items[r['key']]; m = x['cam_to_video']; DT = ev.generated_dt(clip, x['fps'], CLOCK); kc = KC.get(r['key'], 0); cf = x.get('condition_frame_index', 4)
        sa = np.array([m['scale'][0], m['scale'][1]] * 2); sb_ = np.array([m['offset'][0], m['offset'][1]] * 2)
        for n in ev.usable(clip):
            if n not in OBS.get(r['key'], {}) or n not in boxes: continue
            k, ob = OBS[r['key']][n]
            if kc not in set(k.tolist()): continue
            i0 = int(np.where(k == kc)[0][0])
            if np.isnan(ob[i0, 0]): continue
            s = cf + (k - kc) * DT / dt; h = int(hz[n]['horizon']); sel = (s >= cf) & (s <= h)
            o = medfilt3(ob)[sel]; ko = k[sel]; c = (o[:, :2] + o[:, 2:]) / 2; c0 = (ob[i0, :2] + ob[i0, 2:]) / 2
            dobs = np.linalg.norm(c - c0, axis=1); tobs = (ko - kc) * DT
            frac, sb = boxes[n]; F = len(sb); sbv = sb * sa + sb_
            fr = [f for f in range(cf, min(h, F - 1) + 1) if frac[f] >= ev.MIN_VIS]
            if len(fr) < 6 or frac[cf] < ev.MIN_VIS: continue
            cs = (sbv[fr, :2] + sbv[fr, 2:]) / 2; dsim = np.linalg.norm(cs - cs[0], axis=1); tsim = (np.array(fr) - cf) * dt
            okc = np.isfinite(c).all(1); dev, prog = path_stats(c[okc] if okc.any() else c[:0], cs)
            ds = cs[-1] - cs[0]; Ls = float(np.linalg.norm(ds)); r_dir = None
            if Ls >= 20:
                ok_t = okc & (tobs <= tsim[-1] + 1e-6)
                dv = (c[ok_t][-1] - c0) if ok_t.any() else np.zeros(2)
                r_dir = round(float(np.dot(dv, ds) / Ls ** 2), 4)
            out.append(dict(key=r['key'], model=r['model'], clip=clip, obj=n, kind=r['kind'], best=r.get('best'), truth=r.get('truth'),
                            max_obs=float(np.nanmax(dobs)) if np.isfinite(dobs).any() else None, max_sim=float(dsim.max()),
                            p_obs=fit_p(tobs, dobs), p_sim=fit_p(tsim, dsim), t50_obs=t_half(tobs, dobs), t50_sim=t_half(tsim, dsim),
                            dev=dev, prog=prog, r_dir=r_dir, sim_disp=round(Ls, 2), curve_obs=thin(tobs, dobs), curve_sim=thin(tsim, dsim), window_s=float((h - cf) * dt)))
    return out


with get_context('fork').Pool(WORKERS) as pool:
    res = [o for xs in pool.map(run_clip, sorted(by_clip), chunksize=1) for o in xs]
with open(OUT, 'w') as f:
    for o in res: f.write(json.dumps(o) + '\n')
print('written', OUT, len(res), 'objects from', len({o['key'] for o in res}), 'videos')
