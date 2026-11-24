"""Main table, all scenes in one: one row per model, the four prompt groups pooled.
P: concordant-pair share across scenes (per model and parameter, every pair of readable results under the reference-clip reading
with different prompted values; results without a reading do not enter; caution: the mu and v grids are centred on each clip's value,
so a value drawn at random inside the grid already keeps much of the cross-scene order; compare with the grid null); the within-scene
share is kept in the JSON: concordant-pair share of g and v under the main reading (every video under its scene's reference clip; every pair of videos of
one model, scene and prompt group with different prompted values: concordant 1, discordant 0, no order 0.5, i.e. one side without a
moving reading or equal readings; the (C, D, T) units of event_compare.py, every_pair, summed over the groups) and of mu (mu_ref.py,
every_pair, set 'all'); task-balanced median |best / prompted - 1| over the point and interval results (n/a when a model has none),
and the share of results that give such an error. The best value of a result that no library value explains is not used: it sits at
the edge of the grid and measures the grid, not the model. V: ADE ratio, S-IoU, ST-IoU, PSNR, SSIM (common_metrics.py rows) and the share of
videos whose camera drift exceeds 2 % of the image diagonal (cam_comp.py statistics). Reference rows: physically correct videos (the
frame-0 known-truth sets s30f121_f0 and s15v_f0: pairs of clips of one scene with different true values, errors from their results,
V and camera drift from their common_metrics rows and cam_comp.py statistics) and the condition frame held still (V only). The table shows one number per cell; OUT.json keeps C / D, the 95 % intervals and the explained share.
Inputs, with several work dirs: each clip is read from one work dir, by --owner (JSON {"<clip>": dir, "<clip>|concat": dir}) or
else the first listed work dir that has the clip; "<clip>|synthetic" names the --synthetic dir of the clip's re-encoded videos.
Usage: python main_readme_all.py OUT_DIR --groups MAIN_all_clips.json --pooled MAIN_merged_all_clips.json --mu MAIN_MU.json
  --common COMMON.jsonl WORK [WORK ...] [--owner OWNER.json] [--synthetic DIR ...]   -> OUT_DIR/MAIN_README_ALL.md and .json"""
import argparse, collections, glob, itertools, json, math, os, random, re, statistics as st

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('works', nargs='+')
ap.add_argument('--groups', required=True, help='event_compare.py JSON with one group per prompt group (reference_order units)')
ap.add_argument('--pooled', required=True, help='event_compare.py JSON with the prompt groups pooled into one group "all" (errors)')
ap.add_argument('--mu', required=True, help='mu_ref.py JSON'); ap.add_argument('--common', required=True, help='common_metrics.py rows of every scored video, one file')
ap.add_argument('--censored', nargs='*', default=[], help='model:param cells without an explained result: the error is given as a lower bound, '
                'the median over the not-explained results of min(|grid low / prompted - 1|, |grid high / prompted - 1|), shown as ">= X%%"')
ap.add_argument('--exclude-keys', help='JSON list of result keys left out (as event_compare.py)'); ap.add_argument('--results', help='pooled gated results (for --censored)')
ap.add_argument('--view', help='pooled reference-clip view results (event_view.py) for the cross-scene concordance')
ap.add_argument('--mu-readings', nargs='*', default=[], help='WORK:SET of mu_ref.py reference readings (scenes whose mu varies)')
ap.add_argument('--kin', help='kinematics.py rows of every scored video (displacement along the simulated direction, causal order, acceleration exponent)')
ap.add_argument('--exclusions', help='audit trunc_exclusions.json (F110 A2): group_pairs = the same video under two prompt groups, counted once; clip_pairs = '
               'pairs whose parameter value the prompt lost to truncation, left out of that parameter')
ap.add_argument('--keep-keys', help='JSON list of result keys (MODEL.CLIP.GROUP) kept; every other generated video is left out everywhere (2026-11-11: the human-annotated measurable videos)')
ap.add_argument('--drop-clips', default='', help='comma-separated clips left out everywhere (F110 A5: a base row on the task set of its fine-tuned runs)')
ap.add_argument('--owner', help='JSON clip -> work dir'); ap.add_argument('--synthetic', nargs='*', default=[], help='dirs with results_gen/common_synth_*_f0.jsonl (default: the work dirs)')
a = ap.parse_args()
MODELS = {'cogvideox_i2v': 'CogVideoX-1.5', 'cosmos': 'Cosmos-Predict2', 'cosmos3_nano': 'Cosmos3-Nano', 'hunyuan15_i2v': 'HunyuanVideo-1.5',
          'minimax_h3': 'MiniMax-H3', 'physalign_wan22_i2v_a14b': 'PhysAlign', 'physrvg_ti2v5b': 'PhysRVG', 'videogpa_ti2v5b': 'VideoGPA',
          'wan22_i2v_a14b': 'Wan2.2-I2V-A14B', 'wan_ti2v5b': 'Wan2.2-TI2V-5B'}
GROUPS = ('short', 'medium', 'detailed', 'concat')
OWN = json.load(open(a.owner)) if a.owner else {}
def jl(f): return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []
def med(x): return st.median(x) if x else None


def owner(clip, concat, have):
    k = f'{clip}|concat' if concat else clip
    if k in OWN: return OWN[k]
    return next((w for w in a.works if (w, clip) in have), a.works[0] if len(a.works) == 1 else None)   # one evaluation base: every clip is in it


def pooled(units, draws=2000):   # units (C, D, T): concordant 1, discordant 0, no order 0.5
    units = [tuple(u) + (0,) * (3 - len(u)) for u in units]; N = sum(sum(u) for u in units)
    if not N: return None
    sh = lambda us: (sum(u[0] for u in us) + 0.5 * sum(u[2] for u in us)) / sum(sum(u) for u in us)
    rng = random.Random(0); bs = sorted(sh(rng.choices(units, k=len(units))) for _ in range(draws))
    return dict(share=sh(units), ci=[bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]], C=sum(u[0] for u in units), D=sum(u[1] for u in units), T=sum(u[2] for u in units))


G = json.load(open(a.groups)); M = json.load(open(a.pooled)); MU = json.load(open(a.mu)); RO = G['reference_order']
AMP_RATIO = 2.5   # F109 / F110 A3 (2026-11-10 evening): a result whose best-fit simulation moves the fitted objects 2.5 times as far as the video does, or the
def amp_short(paths):   # video 2.5 times as far as the simulation (either at least 10 px), does not explain the video and counts as not explained - points, intervals and bounds
    out = {}   # a bound only when the video's fitted objects move >= 10 px: a still video beyond the grid edge is a true one-sided result (nothing happened), kept
    for f in paths:
        for o in jl(f):
            fb, fo_ = o.get('fit_best_px'), o.get('fit_obs_px')
            if fb is not None and fo_ is not None and math.isfinite(fb) and math.isfinite(fo_) and max(fb, fo_) >= 10 and max(fb, fo_) > AMP_RATIO * max(min(fb, fo_), 1e-9):
                out[(o['key'], o['param'])] = fo_
    return out
def still_set(paths):   # F110 A4: the video's fitted objects move < 10 px while the prompted simulation moves >= 10 px - nothing happened
    return {(o['key'], o['param']) for f in paths for o in jl(f) if o.get('fit_obs_px') is not None and math.isfinite(o['fit_obs_px']) and o['fit_obs_px'] < 10
            and o.get('max_sim_px') is not None and math.isfinite(o['max_sim_px']) and o['max_sim_px'] >= 10}
TABLE_AMP = os.environ.get('PHYSPROBE_TABLE_AMP', '0') == '1'   # F117 (2026-11-11, Codex review): the evaluator checks the amplitude itself (F110 B2, any
# value of the band at its own time shift, every set); this table-level check of the best value without shift could undo its verdict, and
# the physically correct row never had it - off by default
NE_ANY = os.environ.get('PHYSPROBE_NE_ANY', '1') == '1'   # F121 (2026-11-11): a result with a tracked object that no value explains is not
# explained, whatever the other objects fit (the objects the instrument rejects on the physically correct render, D68, aside): the other
# objects alone gave readings like T10 g = 16 for 1.62 from 10 to 18 frames of a pen and a tape roll, while the falling board fitted nothing
def ne_any(r):
    if not NE_ANY or r.get('kind') not in ('point', 'interval', 'upper_bound', 'lower_bound', 'unidentifiable'): return False
    ex = set(r.get('instrument_excluded') or [])
    return any(d.get('status') == 'not_explained' for n, d in (r.get('object_diagnostics') or {}).items() if n not in ex)
def ne_demote(r): return dict(r, kind='not_explained', kind_before_f121=r['kind']) if ne_any(r) else r
def demote(r, flags, key=None):
    r = ne_demote(r)
    if not TABLE_AMP: return r
    fo_ = flags.get(((key or r['key']), r.get('param')))
    if fo_ is None or r.get('kind') not in ('point', 'interval', 'upper_bound', 'lower_bound'): return r
    if r['kind'] in ('upper_bound', 'lower_bound') and fo_ < 10: return r
    return dict(r, kind='not_explained', kind_before_f109=r['kind'])
RD = os.path.dirname(a.results) if a.results else None; RN = os.path.basename(a.results)[:-len('_gated.jsonl')] if a.results else None
AMP_OWN = amp_short([f'{RD}/{RN}_gated_motion.jsonl']) if RD else {}; AMP_REF = amp_short([f'{RD}/{RN}_ref_motion.jsonl']) if RD else {}
STILL_OWN = still_set([f'{RD}/{RN}_gated_motion.jsonl']) if RD else set()
EXJ = json.load(open(a.exclusions)) if a.exclusions and os.path.exists(a.exclusions) else dict(group_pairs=[], clip_pairs=[])
DUP = {b for _, b in EXJ['group_pairs']}   # F110 A2: the second key of a pair of identical videos (medium kept)
TRUNC = {(k, q) for x, y, q in EXJ['clip_pairs'] for k in (x, y)}   # the prompt lost this parameter's value
DUP_TASKS = {'T05_C04'}   # F115 (2026-11-11): the same task as T05_C03 (condition image, prompts and truths identical), counted once
DROP = {c for c in a.drop_clips.split(',') if c} | DUP_TASKS
_SR = next((f'{w}/static_rule.json' for w in a.works if os.path.exists(f'{w}/static_rule.json')), None)
STATIC_REF = set(json.load(open(_SR))['clips']) if _SR else set()   # F110 (2026-11-10): clips whose reference does not move (D-S1, T09_C05): a g or v read through another
# clip's library puts any still video at 0.5-2 and so orders every pair against it; these clips are scored only by the static rule
def mu_undeclared(r):   # F115 (2026-11-11): a concat prompt that declares only one side of the measured contact does not state its friction
    return r.get('param') == 'mu' and str(r.get('key', '')).endswith('.concat') and r.get('truth') is not None and r.get('truth_physical') is not None \
        and abs(r['truth'] - r['truth_physical']) > 1e-3
KEEP = set(json.load(open(a.keep_keys))) if a.keep_keys else None
def kk(k): return k not in DUP and (KEEP is None or k in KEEP)   # --keep-keys: only these generated videos
def kept(r, p=None): return kk(r.get('key')) and r.get('clip') not in DROP and (r.get('key'), p or r.get('param')) not in TRUNC and not mu_undeclared(r) \
    and not (r.get('clip') in STATIC_REF and (p or r.get('param')) in ('g', 'v'))
V = [{k: v for k, v in r.items()} for r in jl(a.common) if kk(r.get('key')) and r.get('clip') not in DROP]
for m in sorted({r.get('model') for r in V} - set(MODELS) - {None}): MODELS[m] = m   # a model outside the list keeps its own name
PRESENT = [m for m in MODELS if any(r.get('model') == m for r in V)]   # table rows: the models of this run
stats = {}
for w in a.works:
    for f in glob.glob(f'{w}/sam_gen/*/tracks_camcomp_stats.jsonl'):
        for s in jl(f):
            if s['key'].count('.') >= 2: stats.setdefault(w, {})[s['key']] = s   # keys model.clip.group
have = {(w, k.split('.')[1]) for w, S in stats.items() for k in S}
CAM = {}
for w, S in stats.items():
    for k, s in S.items():
        if owner(k.split('.')[1], k.endswith('.concat'), have) == w: CAM[k] = s
FROZEN = {}
for w in a.works:
    for r in jl(f'{w}/results_gen/common_frozen_f0.jsonl'):
        if r['clip'] not in DROP and owner(r['clip'], False, {(w, r['clip'])} | have) == w: FROZEN.setdefault(r['clip'], r)   # F124: the reference rows on the same task set
SYNV = {}; SYNCAM = {}
for d in a.synthetic or a.works:
    for f in glob.glob(f'{d}/sam_gen/*_f0/tracks_camcomp_stats.jsonl'):
        for x in jl(f): SYNCAM[(d, x['key'])] = x
    for x in jl(f'{d}/results_gen/common_synth_camera.jsonl'): SYNCAM.setdefault((d, x['key']), x)   # evaluation base: the same statistics, kept with the rows
    for f in sorted(glob.glob(f'{d}/results_gen/common_synth_*_f0.jsonl')):
        for r in jl(f):
            if r.get('clip') in DROP: continue   # F115 / F124 (2026-11-11): the physically correct row on the same tasks, dropped clips included (--drop-clips was ignored here)
            if OWN.get(f"{r['clip']}|synthetic", d) == d: SYNV.setdefault(r['key'], dict(r, _dir=d))   # owner key "<clip>|synthetic": the dir whose re-encoded videos match the clip's reference
SYNP = {}
for w in a.works:
    for s in ('s30f121_f0', 's15v_f0'):
        for r in jl(f'{w}/results_gen/{s}.jsonl'):
            if not re.fullmatch(r'T\d\d_C\d\d', r['clip']) or r['clip'] in DROP: continue   # F124: dropped clips too   # F115: T05_C04 counted once; reference variants of the evaluation base (e.g. T02_C02F) are libraries for readings, not tasks
            if r.get('param') and owner(r['clip'], False, {(w, r['clip'])} | have) == w: SYNP.setdefault((r['clip'], r['param'], s), ne_demote(r))   # F121 as well


def drift(s):   # F120 (2026-11-11): the largest displacement of the image corners (a zoom or dolly moves them, not the centre) over the
    # video's own diagonal; statistics without them (older cam_comp.py) fall back to the centre shift over the 854x480 diagonal
    return s.get('max_corner_px', s['max_shift_px']) / (math.hypot(*s['size']) if s.get('size') else math.hypot(854, 480))
def drift_share(keys):   # largest background displacement > 2 % of the diagonal, over the frames the compensation accepts; F116 (2026-11-11): a video
    # whose compensation rejects more than half of its frames counts as drifting too (the camera moved or the scene changed beyond registration)
    fr = [(drift(CAM[k]) > 0.02) or (CAM[k].get('frames') and (CAM[k].get('frames_failed') or 0) / CAM[k]['frames'] > 0.5)
          for k in keys if k in CAM and CAM[k].get('max_shift_px') is not None]
    return sum(fr) / len(fr) if fr else None


def vcols(rs):
    rel = [r['ade'] / r['ade_static'] for r in rs if r.get('ade') is not None and (r.get('ade_static') or 0) > 0.002]
    l2a = [100 * r['ade'] for r in rs if r.get('ade') is not None]   # all objects, % of the image diagonal (until 2026-11-10)
    l2o = [100 * r['l2_mov'] for r in rs if r.get('l2_mov') is not None]   # moving objects, % of the image height, times both videos observe the object
    l2 = [100 * r['l2_fill'] for r in rs if r.get('l2_fill') is not None]   # main table (2026-11-10 evening): every reference-observed time, a lost object held where last tracked
    lost = [r['lost_mov'] for r in rs if r.get('lost_mov') is not None]   # share of the reference-observed times the generated object is lost
    return dict(l2=med(l2) if len(l2) == len(lost) else None, l2_videos=len(l2), l2_observed=med(l2o), lost=float(np.mean(lost)) if lost else None, l2_all=med(l2a), ade_ratio=med(rel), s_iou=med([r['spatial_iou'] for r in rs if r.get('spatial_iou') is not None]),
                st_iou=med([r['spatiotemporal_iou'] for r in rs if r.get('spatiotemporal_iou') is not None]),
                psnr=med([r['video_psnr'] for r in rs if r.get('video_psnr') is not None]), ssim=med([r['video_ssim'] for r in rs if r.get('video_ssim') is not None]))


import numpy as np
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from order_rule import order_sign, order_matrix, span, reading, ranked_errors, risk, risk_curve, COVERAGE   # D69 / D70; risk-coverage 2026-11-10
import order_rule as _OR; _OR.TIER.setdefault('still', 5)   # F110 A4: a still video's result ranks after every other kind in the 10 % coverage error
VAL = ('point', 'interval', 'upper_bound', 'lower_bound')
KIND = {'upper_bound': 1, 'lower_bound': 2}   # D69: 0 = point / interval
def unit(r, e): lo, hi = span(r); return (r['truth'], r['best'], e[0], e[1], lo, hi, 0 if reading(r) is not None else KIND.get(r['kind'], 3 if r['kind'] == 'interval' else 0))   # D71: kinds of the null: 0 reading, 1 / 2 bound, 3 wide interval
EXK = set(json.load(open(a.exclude_keys))) if a.exclude_keys else set()


def readable(r, static):
    return r.get('param') in ('g', 'mu', 'v') and r['kind'] in VAL and r.get('best') and r.get('truth') and not static.get((r['key'], r['param'])) \
        and r['key'] not in EXK and not (r['param'] == 'mu' and r['clip'][:3] in ('T19', 'T20'))


EDGE = {}
def grid(w, clip, p):
    if (w, clip, p) not in EDGE:
        L = [L for L in json.load(open(f'{w}/library_index.json'))['libraries'] if L['param'] == p and clip in L['members']] if w and os.path.exists(f'{w}/library_index.json') else []
        EDGE[(w, clip, p)] = (min(L[0]['values']), max(L[0]['values'])) if L else None
    return EDGE[(w, clip, p)]


NULL_DRAWS, NULL_BOOT = 1000, 200


MIN_SCENE = 3   # F115 (2026-11-11): scene-equal weighting counts only scenes with at least this many results


def cross(groups, draws=1000):
    """Cross-scene concordance above the grid null. groups = {unit: {scene: [(truth, best, grid_lo, grid_hi), ...]}}; within each unit
    (a model) every pair of readable results with different prompted values is concordant when the measured values keep the prompted
    order (equal readings count in neither); D69: a one-sided bound orders by the range it allows, only against non-overlapping
    ranges (order_rule.py), in the observed share and in the null alike; share = C / (C + D). Null: every result replaced by a log-uniform random value inside its
    own library grid (NULL_DRAWS draws, mean share), because the mu and v grids are centred on each clip's value and a random value
    inside them already keeps much of the cross-scene order. Excess = share - null; 95 % interval of the excess from resampling scenes,
    the null recomputed in every resample (with NULL_BOOT of the draws)."""
    scenes = sorted({sc for g in groups.values() for sc in g})
    rng = np.random.default_rng(0)
    data = {}
    for u, g in groups.items():
        for sc, xs in g.items():
            if not xs: continue
            t = np.array([x[0] for x in xs]); lo = np.log([x[2] for x in xs]); hi = np.log([x[3] for x in xs])
            olo = np.array([x[4] for x in xs]); ohi = np.array([x[5] for x in xs]); k = np.array([x[6] for x in xs])
            z = np.exp(lo[:, None] + (hi - lo)[:, None] * rng.random((len(xs), NULL_DRAWS)))
            zlo = np.where((k == 1)[:, None], 0.0, z); zhi = np.where((k == 2)[:, None], np.inf, z)   # a bound's null keeps its one-sided form, open as in the
            # observation (order_rule.span: upper (0, b], lower [a, inf)); F110 A1 (2026-11-10): it was closed at the grid edge, which the observation never is
            w = np.sqrt(np.where(k == 3, ohi / np.where(olo > 0, olo, 1), 1.0))[:, None]   # D71: a wide interval's null keeps its width
            zlo = np.where((k == 3)[:, None], z / w, zlo); zhi = np.where((k == 3)[:, None], z * w, zhi)
            data[(u, sc)] = (t, (olo, ohi), (zlo, zhi), (lo, hi))
    def cd(pick, k):   # F115 (2026-11-11): every scene weighs the same - a result weighs 1 / (results of its scene in the unit), a pair the product;
        # scenes with fewer than MIN_SCENE results in the unit are left out (a single result would carry its scene's whole weight)
        C = D = 0; CW = DW = 0.0; CN = np.zeros(k); DN = np.zeros(k)
        for u in groups:
            parts = [data[(u, sc)] for sc in pick if (u, sc) in data and len(data[(u, sc)][0]) >= MIN_SCENE]
            if not parts: continue
            t = np.concatenate([q[0] for q in parts]); blo = np.concatenate([q[1][0] for q in parts]); bhi = np.concatenate([q[1][1] for q in parts])
            zlo = np.concatenate([q[2][0][:, :k] for q in parts]); zhi = np.concatenate([q[2][1][:, :k] for q in parts])
            if len(t) < 2: continue
            iu = np.triu_indices(len(t), 1); st_ = np.sign(t[:, None] - t[None, :])[iu]
            glo = np.concatenate([q[3][0] for q in parts]); ghi = np.concatenate([q[3][1] for q in parts])
            wt = np.concatenate([np.full(len(q[0]), 1.0 / len(q[0])) for q in parts])
            ov = (glo[iu[0]] < ghi[iu[1]]) & (glo[iu[1]] < ghi[iu[0]])   # 2026-11-10: only pairs whose library grids overlap (otherwise any video keeps the order)
            keep = (st_ != 0) & ov; st_ = st_[keep]
            v = st_ * order_matrix(blo, bhi)[iu][keep]; C += int((v > 0).sum()); D += int((v < 0).sum())
            a, b_ = iu[0][keep], iu[1][keep]; ww = wt[a] * wt[b_]; CW += float(ww[v > 0].sum()); DW += float(ww[v < 0].sum())
            zz = ((zlo[a] > zhi[b_]).astype(int) - (zhi[a] < zlo[b_]).astype(int)) * st_[:, None]
            CN += ((zz > 0) * ww[:, None]).sum(0); DN += ((zz < 0) * ww[:, None]).sum(0)
        return C, D, CW, DW, CN, DN
    C, D, CW, DW, CN, DN = cd(scenes, NULL_DRAWS)
    if not C + D: return None
    share = CW / (CW + DW)
    def nmean(cn, dn):   # F118 (2026-11-11, Codex review): a draw in which no pair is ordered says nothing about chance - left out (it scored 0)
        ok = (cn + dn) > 0
        return float(np.mean(cn[ok] / (cn[ok] + dn[ok]))) if ok.any() else None
    null = nmean(CN, DN)
    if null is None: return None
    r = random.Random(0); bs = []
    for _ in range(draws):
        pick = r.choices(scenes, k=len(scenes)); c, d, cw, dw, cn, dn = cd(pick, NULL_BOOT)
        nb = nmean(cn, dn) if c + d else None
        if nb is not None: bs.append(cw / (cw + dw) - nb)
    bs.sort()
    nz = (share - null) / null if share < null else None if null > 0.95 else (share - null) / (1 - null)   # 0 = chance, 1 = every pair in order, -1 = every pair reversed (2026-11-10); above a chance level over 0.95 there is no room left: not shown (below it the scale is the chance level itself)
    return dict(share=share, null=null, excess=share - null, normalized=nz,
                ci=[bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]], C=C, D=D, scenes=len(scenes))


XS = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))   # param -> model -> scene -> [(truth, best)]
if a.view:
    VW = jl(a.view); VM = {(o['key'], o['param']): o for o in jl(a.view.replace('.jsonl', '_motion.jsonl'))}
    static = {k: bool(o.get('static')) for k, o in VM.items()}
    MUR = {}   # mu reading under the scene's reference clip, by the video's own key
    for spec in a.mu_readings:
        w, st_name = spec.split(':'); IT = {x['key']: x for x in json.load(open(f'{w}/sam_gen/{st_name}/items.json'))['items']}
        mo = {o['key']: bool(o.get('static')) for o in jl(f'{w}/results_gen/{st_name}_motion.jsonl') if o.get('param') == 'mu'}
        AMP_MU = amp_short([f'{w}/results_gen/{st_name}_motion.jsonl'])
        for r in jl(f'{w}/results_gen/{st_name}_gated.jsonl'):
            if r.get('param') == 'mu' and r['key'] in IT: MUR.setdefault(IT[r['key']]['key_own'], dict(demote(r, AMP_MU), _static=mo.get(r['key'])))
    for r in VW:
        if r.get('kind') == 'not_validated' or r.get('model') not in MODELS or not kept(r): continue
        rc = r.get('ref_clip'); r = demote(r, AMP_REF, f"{r['key']}.ref.{rc}") if rc and rc != r['clip'] else demote(r, AMP_OWN)
        gclip = r.get('ref_lib') or r.get('ref_clip') or r.get('clip')   # the library the reading comes from
        if r.get('param') == 'mu' and r['key'] in MUR:
            x = MUR[r['key']]; gclip = x['clip']; r = dict(x, key=r['key'], clip=r['clip'], truth=r['truth'], model=r['model']); static[(r['key'], 'mu')] = x['_static']
        if readable(r, static):
            e = grid(owner(gclip, r['key'].endswith('.concat'), have), gclip, r['param'])
            if e: XS[r['param']][r['model']][r['clip'][:3]].append(unit(r, e))
CROSS = {p: {m: cross({m: XS[p][m]}) for m in MODELS} for p in ('g', 'mu', 'v')}
for p in ('g', 'mu', 'v'): CROSS[p]['all'] = cross({m: XS[p][m] for m in MODELS})


OUT = {}
for m in list(MODELS) + ['all']:
    cc = {p: pooled([tuple(u) for g in GROUPS for u in RO.get(g, {}).get(m, {}).get(p, {}).get('all', {}).get('every_pair', {}).get('units', [])]) for p in ('g', 'v')}
    cc['mu'] = pooled([tuple(u) for u in (MU.get(f'all|{m}', {}).get('every_pair') or {}).get('units', [])])
    s = M['stats']['all'].get(m) or {}
    rs = [r for r in V if m == 'all' or r.get('model') == m]
    ne, nr = sum(s.get(f'n_err_{p}', 0) for p in ('g', 'mu', 'v')), sum(s.get(f'n_res_{p}', 0) for p in ('g', 'mu', 'v'))
    OUT[m] = dict(concordant_within_scene=cc, concordant={p: CROSS[p][m] for p in ('g', 'v', 'mu')}, err={p: s.get(f'err_any_{p}') for p in ('g', 'mu', 'v')}, tasks={p: s.get(f'tasks_{p}') for p in ('g', 'mu', 'v')},
                  n_err={p: s.get(f'n_err_{p}') for p in ('g', 'mu', 'v')}, n_res={p: s.get(f'n_res_{p}') for p in ('g', 'mu', 'v')},
                  err_share=ne / nr if nr else None,
                  explained=s.get('explained'), videos=len(rs), drift_gt2=drift_share({r['key'] for r in rs}), **vcols(rs))
CELLS = a.censored or ([f'{m}:{p}' for m in PRESENT for p in ('g', 'mu', 'v') if OUT[m]['err'][p] is None] if a.results else [])   # default: every cell without an explained result
for cell in CELLS:   # a not-explained result lies outside the grid, so its error is at least the distance to the nearer grid edge
    m, p = cell.split(':'); lb = []
    for r in jl(a.results):
        if r.get('model') != m or r.get('param') != p or r['kind'] != 'not_explained' or r['key'] in EXK or not r.get('truth'): continue
        e = grid(owner(r['clip'], r['key'].endswith('.concat'), have), r['clip'], p)
        if e: lb.append(100 * min(abs(e[0] / r['truth'] - 1), abs(e[1] / r['truth'] - 1)))
    if lb and OUT[m]['err'][p] is None: OUT[m].setdefault('err_lower_bound', {})[p] = dict(median=med(lb), results=len(lb))
def onset(c, q=0.2, floor=3.0):   # first time an object reaches max(20 % of its largest displacement, 3 px)
    c = np.asarray(c); m_ = c[:, 1].max(); th = max(q * m_, floor)
    return None if m_ < th else float(c[np.argmax(c[:, 1] >= th), 0])


def sim_pairs(v):   # (earlier, later) object pairs the simulation orders: simulated onsets >= 0.3 s apart, both simulated >= 15 px
    out = []
    for x, y in itertools.combinations([r for r in v if r['max_sim'] >= 15 and r.get('curve_sim')], 2):
        sx, sy = onset(x['curve_sim']), onset(y['curve_sim'])
        if sx is None or sy is None or abs(sx - sy) < 0.3: continue
        out.append((x, y) if sx < sy else (y, x))
    return out
def pair_kept(x, y):   # 1 the video keeps the order by >= 0.1 s, 0 it does not, None it does not let the pair in (an object < 15 px or no onset)
    if (x['max_obs'] or 0) < 15 or (y['max_obs'] or 0) < 15 or not x.get('curve_obs') or not y.get('curve_obs'): return None
    ox, oy = onset(x['curve_obs']), onset(y['curve_obs'])
    return None if ox is None or oy is None else int(oy - ox >= 0.1)
def kin_design(syn_rows):
    """F123 (2026-11-11, shuan: one denominator, and the physically correct row near 1 and 2): the pairs and objects that the physically
    correct video of the clip lets in, in every re-encoded format - the instrument can measure them; None without physically correct rows"""
    if not syn_rows: return None
    QP = QO = None
    for fmt in sorted({r.get('model') for r in syn_rows}):
        by = collections.defaultdict(list)
        for r in syn_rows:
            if r.get('model') == fmt: by[r['key']].append(r)
        qp = {(v[0]['clip'], x['obj'], y['obj']) for v in by.values() for x, y in sim_pairs(v) if pair_kept(x, y) is not None}
        qo = {(r['clip'], r['obj']) for v in by.values() for r in v if (r.get('p_sim') or 0) >= 1.6 and r.get('p_obs') is not None}
        QP = qp if QP is None else QP & qp; QO = qo if QO is None else QO & qo
    return QP, QO


def kin_cols(rows):
    """three kinematic columns (2026-11-10): median displacement along the simulated direction (objects the simulation moves >= 20 px),
    share of object pairs whose simulated onsets differ by >= 0.3 s that keep the order by >= 0.1 s, median acceleration exponent p of the
    objects the simulation accelerates (p_sim >= 1.6). F123 (2026-11-11): the order and the exponent are taken over the design (KDES:
    the pairs and objects the physically correct videos let in), one denominator for every model; a pair whose objects do not both move
    >= 15 px with an onset counts as not kept, an object without fitted motion as p = 0 (no displacement). Identical outputs of one model
    under two prompt groups count once."""
    by = collections.defaultdict(list)
    for r in rows: by[r['key']].append(r)
    V = [v for k, v in sorted(by.items())]   # F118 (2026-11-11, Codex review): identical videos are already left out by the audit (DUP); the old
    # signature (clip, largest displacement, curve length) also merged different videos
    rs = [r for v in V for r in v]
    rd = [r['r_dir'] for r in rs if r.get('r_dir') is not None]
    if KDES is None:   # no physically correct rows: the old rules (only what the video lets in)
        pp = [r['p_obs'] for r in rs if (r.get('p_sim') or 0) >= 1.6 and r.get('p_obs') is not None]
        ks = [(k_, v[0]['key']) for v in V for x, y in sim_pairs(v) for k_ in [pair_kept(x, y)] if k_ is not None]
    else:
        QP, QO = KDES
        pp = [r['p_obs'] if r.get('p_obs') is not None else 0.0 for r in rs if (r.get('p_sim') or 0) >= 1.6 and (r['clip'], r['obj']) in QO]
        ks = [(pair_kept(x, y) or 0, v[0]['key']) for v in V for x, y in sim_pairs(v) if (v[0]['clip'], x['obj'], y['obj']) in QP]
    n = len(ks)
    return dict(r_dir=float(np.median(rd)) if rd else None, n_dir=len(rd), p=float(np.median(pp)) if pp else None, n_p=len(pp),
                order_kept=sum(k_ for k_, _ in ks) / n if n else None, pairs=n, pair_videos=len({key for _, key in ks}))   # F110 A6: support in videos


SYNKIN = []
for d in a.synthetic or a.works:
    for f in sorted(glob.glob(f'{d}/results_gen/kin_synth_*_f0.jsonl')): SYNKIN += [r for r in jl(f) if r.get('clip') not in DROP]   # F124
KDES = kin_design(SYNKIN)   # F123
KIN = [r for r in jl(a.kin) if kk(r.get('key')) and r.get('clip') not in DROP] if a.kin else []
for m in PRESENT + ['all']:
    if m in OUT and KIN: OUT[m]['kinematics'] = kin_cols([r for r in KIN if (r.get('model') in PRESENT if m == 'all' else r.get('model') == m)])


def scene_risk(rows, c=COVERAGE):
    """F115 (2026-11-11): every scene weighs the same - the most confident share c of each scene's results (at least one, scenes
    with at least MIN_SCENE results), pooled with weight 1 / k; slots without a measurable result count as 100 % error (F117);
    weighted median of |value / prompted - 1|; None when no scene qualifies"""
    by = collections.defaultdict(list)
    for r in rows: by[r['clip'][:3]].append(r)
    vals, wts = [], []
    for rs in by.values():
        if len(rs) < MIN_SCENE: continue
        k = max(1, int(math.floor(c * len(rs) + 0.5 + 1e-9)))   # F118: half up, the same k for 0.1 and the grid's 0.09999999999999999
        e = ranked_errors(rs)[:k]; e = e + [1.0] * (k - len(e))   # F117 (2026-11-11): every scene fills its k slots; a slot without a measurable
        vals += e; wts += [1.0 / k] * k                            # result counts as 100 % error (the parameter's effect absent, e.g. g = 0)
    if not vals: return None
    o = np.argsort(vals); v = np.asarray(vals)[o]; cw = np.cumsum(np.asarray(wts)[o])
    return float(v[np.searchsorted(cw, cw[-1] / 2)])
def cov_stats(rs):   # risk-coverage (2026-11-10): error of the most confident COVERAGE share of all results, readings, the curve
    out = {}
    for p in ('g', 'mu', 'v'):
        x = [r for r in rs if r.get('param') == p and r.get('truth') and r.get('kind') != 'not_validated' and not (p == 'mu' and r['clip'][:3] in ('T19', 'T20'))]   # F110 A7
        xs = [dict(r, kind='still') if (r['key'], p) in STILL_OWN and reading(r) is None else r for r in x]   # F110 A4: still videos rank last
        e = ranked_errors(xs)
        out[p] = dict(results=len(x), readings=sum(1 for r in x if reading(r) is not None), with_value=len(e), err=scene_risk(xs),
                      curve=[(round(float(c), 2), scene_risk(xs, round(float(c), 2))) for c in np.arange(0.01, 1.0001, 0.01)], err_pooled=risk(e, len(x)))   # F115: scene-equal
    return out
GR = [demote(r, AMP_OWN) for r in jl(a.results) if r.get('key') not in EXK and r.get('model') and kept(r)] if a.results else []
for m in PRESENT + ['all']:
    if m in OUT: OUT[m]['coverage'] = cov_stats([r for r in GR if (r['model'] in PRESENT if m == 'all' else r['model'] == m)])
rows = [r for (c, p, _), r in SYNP.items() if not (p == 'mu' and c[:3] in ('T19', 'T20'))]
def syn_cross(p):   # physically correct videos, the same cross-scene pairs (both re-encoded formats pooled)
    g = collections.defaultdict(list)
    for (c, q, _), r in SYNP.items():
        if q == p and r['kind'] in VAL and r.get('best') and r.get('truth') and not (p == 'mu' and c[:3] in ('T19', 'T20')):
            e = grid(owner(c, False, have), c, p)
            if e: g[c[:3]].append(unit(r, e))
    return cross({'syn': g})
def syn_pairs(p):   # physically correct videos: pairs of clips of one scene with different true values, per set; ties 0.5 as above
    grp = {}
    for (c, q, sset), r in SYNP.items():
        if q == p and r.get('truth') and not (p == 'mu' and c[:3] in ('T19', 'T20')): grp.setdefault((sset, c[:3]), []).append(r)
    U = []
    for rs in grp.values():
        cc = dd = tt = 0
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                x, y = rs[i], rs[j]
                if round(x['truth'], 6) == round(y['truth'], 6): continue
                ok = all(z['kind'] in VAL and z.get('best') for z in (x, y)); s_ = (x['truth'] - y['truth']) * order_sign(x, y) if ok else 0   # D69
                cc += s_ > 0; dd += s_ < 0; tt += s_ == 0
        if cc + dd + tt: U.append((cc, dd, tt))
    return pooled(U)
syn_drift = [drift(SYNCAM[(r['_dir'], r['key'])]) for r in SYNV.values() if (r['_dir'], r['key']) in SYNCAM]
syn_ne = sum(1 for r in rows if reading(r) is not None and r.get('truth'))   # D70
syn = dict(err_share=syn_ne / len(rows) if rows else None, concordant={p: syn_cross(p) for p in ('g', 'v', 'mu')}, concordant_within_scene={p: syn_pairs(p) for p in ('g', 'v', 'mu')}, drift_gt2=sum(x > 0.02 for x in syn_drift) / len(syn_drift) if syn_drift else None, drift_videos=len(syn_drift),
           err={p: med([100 * abs(reading(r) / r['truth'] - 1) for r in rows if r['param'] == p and reading(r) is not None and r.get('truth')]) for p in ('g', 'mu', 'v')},   # D70
           tasks={p: len({r['clip'] for r in rows if r['param'] == p and r.get('best') is not None}) for p in ('g', 'mu', 'v')},
           explained=sum(r['kind'] not in ('not_explained', 'no_observation', 'static_moved') for r in rows) / len(rows) if rows else None,
           videos=len(SYNV), v_clips=len({r['clip'] for r in SYNV.values()}), **vcols(list(SYNV.values())))
syn['coverage'] = cov_stats(rows); syn['kinematics'] = kin_cols(SYNKIN) if SYNKIN else {}; OUT['physically_correct'] = syn; OUT['held_still'] = dict(clips=len(FROZEN), **vcols(list(FROZEN.values())))
os.makedirs(a.out, exist_ok=True); json.dump(OUT, open(f'{a.out}/MAIN_README_ALL.json', 'w'), indent=1, default=float)
c2 = lambda d: '0.00*' if not d or d.get('normalized') is None else f"{d['normalized']:+.2f}" + ('*' if d.get('C', 0) + d.get('D', 0) < 30 else '')   # normalized concordance (2026-11-10); * = fewer than 30 orderable pairs, or none above chance: shown at chance
pe = lambda x: 'n/a' if x is None else f'{x:.1f}%'
sh = lambda x: '0%' if not x else f'{100 * x:.0f}%'
dr = lambda x: '–' if x is None else f'{100 * x:.0f}%'
nf = lambda x, k: '–' if x is None else format(x, k)
L = ['# Main table: all scenes', '', '| model | g concordance ↑ | v concordance ↑ | μ concordance ↑ | g error ↓ | μ error ↓ | v error ↓ | readings g / μ / v | displacement along the simulated direction (1 = right) | causal order kept ↑ | acceleration exponent (2 = right) | L2 trajectory error ↓ | tracking lost ↓ | S-IoU ↑ | ST-IoU ↑ | PSNR ↑ | SSIM ↑ | camera drift > 2 % ↓ |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
kcol = lambda k: f"{nf(k.get('r_dir'), '.2f')} | {dr(k.get('order_kept'))} | {nf(k.get('p'), '.2f')}"
for m in PRESENT + (['all'] if len(PRESENT) > 1 else []):
    o = OUT[m]
    cv = o.get('coverage', {}); ec = lambda p: pe(None if (cv.get(p) or {}).get('err') is None else 100 * cv[p]['err'])   # error at COVERAGE (2026-11-10)
    rd = ' / '.join(str((cv.get(p) or {}).get('readings', '–')) for p in ('g', 'mu', 'v'))
    kc = kcol(o.get('kinematics') or {})
    L.append(f"| {MODELS.get(m, 'All models')} | {c2(o['concordant']['g'])} | {c2(o['concordant']['v'])} | {c2(o['concordant']['mu'])} | {ec('g')} | {ec('mu')} | {ec('v')} | {rd} | {kc} | "
             f"{nf(o['l2'], '.2f')}% | {dr(o['lost'])} | {nf(o['s_iou'], '.3f')} | {nf(o['st_iou'], '.3f')} | {nf(o['psnr'], '.1f')} | {nf(o['ssim'], '.3f')} | {dr(o['drift_gt2'])} |")
sc = lambda p: pe(None if syn['coverage'][p]['err'] is None else 100 * syn['coverage'][p]['err'])
L.append(f"| Physically correct videos | {c2(syn['concordant']['g'])} | {c2(syn['concordant']['v'])} | {c2(syn['concordant']['mu'])} | {sc('g')} | {sc('mu')} | {sc('v')} | {' / '.join(str(syn['coverage'][p]['readings']) for p in ('g', 'mu', 'v'))} | {kcol(syn['kinematics'])} | {nf(syn['l2'], '.2f')}% | {dr(syn['lost'])} | {nf(syn['s_iou'], '.3f')} | {nf(syn['st_iou'], '.3f')} | {nf(syn['psnr'], '.1f')} | {nf(syn['ssim'], '.3f')} | {dr(syn['drift_gt2'])} |")
h = OUT['held_still']
L.append(f"| Condition frame held still | V reference only | | | | | | | 0.00 | – | – | {nf(h['l2'], '.2f')}% | {dr(h['lost'])} | {nf(h['s_iou'], '.3f')} | {nf(h['st_iou'], '.3f')} | {nf(h['psnr'], '.1f')} | {nf(h['ssim'], '.3f')} | – |")
L += ['', f"Physically correct videos: P errors over {syn['tasks']['g']} tasks (g), V over {syn['v_clips']} tasks; condition frame held still: {h['clips']} tasks. "
      'Concordance: per model and parameter, every pair of readable results (reference-clip reading) with different prompted values whose library grids overlap, across scenes and prompt groups (results without a reading do not enter; pairs on disjoint grids keep the order whatever the video shows); share of pairs whose measured values keep the prompted order, scaled so that 0 = the chance share and 1 = every pair in order (-1 = every pair reversed). Chance share: every result replaced by a log-uniform random value inside its own library grid (mean of 1,000 draws); the g grid is shared by all clips (chance 0.5), the μ and v grids are centred on each clip\'s prompted value, so their chance share is higher. '
      'L2 trajectory error: moving objects only (largest reference displacement >= 4 % of the image height), distance between the tracked box centres of the generated and the reference video (displacement from the condition frame), in % of the image height, mean over every time the reference observes the object (a lost object held at its last tracked position), mean over the moving objects, median over videos. Tracking lost: share of those times at which the generated object is not observed. '
      'Errors (g, μ, v): within every scene with at least three results, the results are ranked by how firmly the evaluator measured them (readings - a point or an interval within a factor 3 - first, then wider intervals, one-sided bounds, unidentifiable and not explained results with their best-fit value, results of still videos last; within a tier the smaller fit residual first); the most confident 10 % of each scene (at least one; a slot without a measurable result counts as 100 %) are pooled with every scene weighing the same, and the error is their weighted median of |value / prompted - 1|. Concordance weighs every scene the same as well.']
open(f'{a.out}/MAIN_README_ALL.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
