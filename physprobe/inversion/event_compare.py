"""Comparison tables of the four-prompt evaluation, per model and prompt group: the four outcomes, task-balanced parameter errors
(median over tasks of the per-task median |best / prompt - 1|, point and interval results, at least 3 tasks), the share of results
whose band contains the prompted value, and the concordant-pair shares. Main reading: every video under its scene's reference clip
(the GROUP's @VIEW results from event_ref.py), pairs of videos of the same model and scene with different prompted values; sets
'all', 'pure' (both clips equal to each other and to the reference in every other parameter) and 'admitted' (pure, and the scene and
parameter passed the known-truth admission, --admission). The own-clip shares (each video under its own clip) are kept in an appendix.
Groups: LABEL=RESULTS[:KEYS][@VIEW], RESULTS = results_gen/RESULTS.jsonl and RESULTS_motion.jsonl in WORK; KEYS = a key prefix, or
*SUFFIX for keys ending with it (e.g. *.short for the keys model.clip.group). Simulator values of every clip (for the pure set and
the consistency check): --truths JSON {clip: {param: value}}, else the library_index.json of WORK and of --truth-from dirs.
Usage: python event_compare.py WORK OUT.md GROUP [GROUP ...] [--tasks common|all|CLIPS.json] [--title TEXT]
  -> OUT.md and OUT.json (stats, concordance, reference_order with the (C, D) units per model and scene)"""
import argparse, collections, json, random, statistics as st
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('work'); ap.add_argument('out'); ap.add_argument('groups', nargs='+')
ap.add_argument('--tasks', default='common', help='common | all | a JSON list of clips (e.g. final_same_video_clips.json)'); ap.add_argument('--params', default='g,mu,v', help='parameters kept (e.g. mu for the E4 delta)'); ap.add_argument('--title', default='Event-prompt comparison'); ap.add_argument('--exclude-keys', help='JSON list of result keys left out (e.g. versions whose model input was identical)')
ap.add_argument('--exclude-pairs', help='trunc_exclusions.json of prompt_trunc_audit.py: cross-clip pairs [key, key, param] the model could not tell apart (difference cut away by truncation, or identical input), left out of the concordant-pair shares')
ap.add_argument('--admission', help='reference_admission.json (admission.py); with it the main order table uses the pure and admitted (scene, parameter) only')
ap.add_argument('--truths', help='JSON {clip: {param: value}} of the simulator values (default: library_index.json of WORK and --truth-from)')
ap.add_argument('--truth-from', nargs='*', default=[], help='further work dirs whose library_index.json gives simulator values')
ap.add_argument('--models', help='comma-separated model names (default: the ten models below)')
a = ap.parse_args()
W = Path(a.work) / 'results_gen'
MODELS = {'cogvideox_i2v': 'CogVideoX-1.5', 'cosmos': 'Cosmos-Predict2', 'cosmos3_nano': 'Cosmos3-Nano', 'hunyuan15_i2v': 'HunyuanVideo-1.5',
          'minimax_h3': 'MiniMax-H3', 'physalign_wan22_i2v_a14b': 'PhysAlign', 'physrvg_ti2v5b': 'PhysRVG', 'videogpa_ti2v5b': 'VideoGPA',
          'wan22_i2v_a14b': 'Wan2.2-I2V-A14B', 'wan_ti2v5b': 'Wan2.2-TI2V-5B'}
if a.models: MODELS = {m: MODELS.get(m, m) for m in a.models.split(',')}
OUT4 = ('did_not_happen', 'unexplained', 'explained', 'no_observation')
GP = {'C01': 1.62, 'C02': 3.71, 'C03': 9.81}


VIEW = {}   # label -> (rows, motion) of the reference-clip view: LABEL=RESULTS[:PRE][@VIEW]


EXCL = set(json.load(open(a.exclude_keys))) if a.exclude_keys else set()
XP = {(frozenset(x[:2]), x[2]) for x in json.load(open(a.exclude_pairs))['clip_pairs']} if a.exclude_pairs else set()
base = lambda k: k.split('.ref.')[0]   # reference-clip rows carry the key of their video plus .ref.<clip>
unseen = lambda x, y, p: (frozenset((base(x['key']), base(y['key']))), p) in XP


def load(spec):
    label, rest = spec.split('=', 1); rest, _, view = rest.partition('@'); name, _, pre = rest.partition(':')
    sel = (lambda k: k.endswith(pre[1:])) if pre.startswith('*') else (lambda k: k.startswith(pre))   # '*SUFFIX': keys ending with it (event keys model.clip.group)
    R = [r for r in map(json.loads, open(W / f'{name}.jsonl')) if r.get('param') in a.params.split(',') and r.get('model') in MODELS and sel(r['key'])]
    if EXCL: R = [r for r in R if r['key'] not in EXCL]
    MO = {(o['key'], o['param']): o for o in map(json.loads, open(W / f'{name}_motion.jsonl')) if sel(o['key'])}
    if view:
        VR = [r for r in map(json.loads, open(W / f'{view}.jsonl')) if r.get('param') in ('g', 'v') and r.get('model') in MODELS and sel(r['key']) and r.get('ref_clip')]
        VM = {(o['key'], o['param']): o for o in map(json.loads, open(W / f'{view}_motion.jsonl')) if sel(o['key'])}
        VIEW[label] = (VR, VM)
    return label, name + ((f' (keys *{pre[1:]})' if pre.startswith('*') else f' (keys {pre}*)') if pre else ''), R, MO


G = [load(s) for s in a.groups]
tasks = [{r['clip'] for r in R} for _, _, R, _ in G]
tasks = [t for t in tasks if t] or [set()]   # a group without results yet (e.g. a dry run of the merge) does not empty the common set
keep = set.intersection(*tasks) if a.tasks == 'common' else (set.union(*tasks) if a.tasks == 'all' else set(json.load(open(a.tasks))) & set.union(*tasks))   # a clip list: only these (and present somewhere)


def outcome(r, MO):
    if r['kind'] == 'no_observation': return 'no_observation'
    if MO.get((r['key'], r['param']), {}).get('static'): return 'did_not_happen'
    return 'unexplained' if r['kind'] in ('not_explained', 'static_moved') else 'explained'   # static references: static_still = explained, static_moved = unexplained


def stats(R, MO):
    """R: results of one model and group (gated rows; mu of T19 / T20 already removed)"""
    s = {'videos': len({r['key'] for r in R}), 'results': len(R)}
    c = collections.Counter(outcome(r, MO) for r in R); n = sum(c.values()) or 1
    s.update({o: c[o] / n for o in OUT4})
    # 95% interval of each outcome share by resampling tasks (1,000 draws, fixed seed): one seed and one prompt per group, so
    # differences between groups need it
    bt = collections.defaultdict(list)
    for r in R: bt[r['clip']].append(outcome(r, MO))
    ks = sorted(bt); rng = random.Random(0); draws = collections.defaultdict(list)
    for _ in range(1000):
        cc = collections.Counter(o for k in rng.choices(ks, k=len(ks)) for o in bt[k]); nn = sum(cc.values()) or 1
        for o in OUT4: draws[o].append(cc[o] / nn)
    for o in OUT4: v = sorted(draws[o]); s[o + '_ci'] = (v[25], v[974])
    for p in ('g', 'mu', 'v'):
        t = collections.defaultdict(list)
        for r in R:
            if r['param'] == p and reading(r) is not None and r.get('truth'): t[r['clip']].append(abs(reading(r) / r['truth'] - 1) * 100)   # D70
        s[f'err_{p}'] = st.median([st.median(v) for v in t.values()]) if len(t) >= 3 else None; s[f'tasks_{p}'] = len(t)
        t2 = collections.defaultdict(list)   # every result with tracked motion and a best value (not explained included; diagnostic only: those best values sit at the grid edges)
        for r in R:
            if r['param'] == p and r.get('best') and r.get('truth') and outcome(r, MO) in ('explained', 'unexplained'): t2[r['clip']].append(abs(r['best'] / r['truth'] - 1) * 100)
        s[f'err_best_{p}'] = st.median([st.median(v) for v in t2.values()]) if len(t2) >= 3 else None; s[f'tasks_best_{p}'] = len(t2)
        # main-table error: point / interval results as err_{p}, without the 3-task minimum; with the share of results that give it
        s[f'err_any_{p}'] = st.median([st.median(v) for v in t.values()]) if t else None
        s[f'n_err_{p}'] = sum(1 for r in R if r['param'] == p and reading(r) is not None and r.get('truth'))   # D70
        s[f'n_res_{p}'] = sum(1 for r in R if r['param'] == p and r.get('truth') is not None)
    inb = [bool(r.get('truth_in_band')) for r in R if r.get('truth') is not None and r.get('band') and r['kind'] in ('point', 'interval', 'lower_bound', 'upper_bound', 'unidentifiable')]
    s['in_band'] = sum(inb) / len(inb) if inb else None
    s['n_band'] = len(inb)
    for c_, gv in GP.items():
        v = [r.get('est') or r.get('best') for r in R if r['param'] == 'g' and r['clip'][4:] == c_ and abs((r.get('truth') or 0) - gv) < 0.05
             and r['kind'] in ('point', 'interval') and outcome(r, MO) == 'explained']
        s[f'g_{c_}'] = (st.median(v), len(v)) if v else (None, 0)
    return s


VAL = ('point', 'interval', 'upper_bound', 'lower_bound')


def concord(R, MO, p, draws=2000):
    """Own-clip concordant-pair share: within each (scene, model), every
    pair of results of parameter p whose prompted values differ is compared on best when both are moving and carry a value
    (kinds point / interval / upper / lower bound): concordant if best follows the prompted order, discordant if reversed, tie if
    equal. Share = C / (C + D), 95% interval from resampling the (scene, model) groups; chance 0.5. Also: ties, comparable /
    possible pairs, static share of the results that have a prompted value."""
    grp = collections.defaultdict(list)
    for r in R:
        if r['param'] == p and r.get('truth') is not None: grp[(r['clip'][:3], r['model'])].append(r)
    per = []; nst = nall = 0
    for rs in grp.values():
        st_ = [bool(MO.get((r['key'], p), {}).get('static')) for r in rs]; nst += sum(st_); nall += len(rs)
        use = [(r['truth'], r['best'], r) for r, s_ in zip(rs, st_) if r['kind'] in VAL and r.get('best') and not s_]
        poss = sum(1 for i in range(len(rs)) for j in range(i + 1, len(rs)) if rs[i]['truth'] != rs[j]['truth'] and not unseen(rs[i], rs[j], p))
        c = d = t = 0
        for i in range(len(use)):
            for j in range(i + 1, len(use)):
                (ti, bi, ri), (tj, bj, rj) = use[i], use[j]
                if ti == tj or unseen(ri, rj, p): continue
                x = (ti - tj) * (bi - bj); c += x > 0; d += x < 0; t += x == 0
        if poss: per.append((c, d, t, poss))
    C, D, Tt, P = (sum(x[k] for x in per) for k in range(4))
    rng = random.Random(0); bs = []
    for _ in range(draws if per else 0):
        smp = rng.choices(per, k=len(per)); cc, dd = sum(x[0] for x in smp), sum(x[1] for x in smp)
        if cc + dd: bs.append(cc / (cc + dd))
    bs.sort()
    return dict(share=C / (C + D) if C + D else None, ci=(bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]) if bs else None,
                C=C, D=D, ties=Tt, comparable=C + D + Tt, possible=P, groups=len(per), static=nst / nall if nall else None, n=nall)


import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import round_truth
from order_rule import order_sign, reading   # D69 / D70
PHYS = collections.defaultdict(dict, round_truth.load(a.truths) if a.truths else round_truth.truth_table([a.work] + a.truth_from))
round_truth.check(PHYS, [r for _, _, R, _ in G for r in R], 'event_compare')
same_other = lambda x, y, p: all(PHYS[x].get(q) == PHYS[y].get(q) for q in ('g', 'mu', 'v') if q != p)
ADM = json.load(open(a.admission)) if a.admission else None


def censored(r):
    lo, hi = r.get('band') or (r.get('best'), r.get('best'))
    return (0.0 if r['kind'] == 'upper_bound' else lo), (float('inf') if r['kind'] == 'lower_bound' else hi)


def pair_score(R, MO, p, mode, draws=2000):
    """Every pair of videos of the same model and scene with different prompted values (under the reference clip): concordant 1,
    discordant 0, no order 0.5 (one or both without a moving reading with a value, or equal readings), as ties count in Kendall's
    tau and the AUC. Share = (C + T / 2) / (C + D + T). Units (C, D, T) per (model, scene); 95 % interval from resampling them."""
    grp = collections.defaultdict(list)
    for r in R:
        if r['param'] != p or not r.get('truth'): continue
        if mode != 'all' and not same_other(r['clip'], r['ref_clip'], p): continue
        if mode == 'admitted' and (not ADM or ADM.get(f"{r['clip'][:3]}|{p}", {}).get('status') != 'pass'): continue
        grp[(r['model'], r['clip'][:3])].append(r)
    read = lambda r: r['kind'] in VAL and r.get('best') and not MO.get((r['key'], p), {}).get('static')
    U = []
    for rs in grp.values():
        c = d = t = 0
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                x, y = rs[i], rs[j]
                if x['truth'] == y['truth'] or unseen(x, y, p) or (mode != 'all' and not same_other(x['clip'], y['clip'], p)): continue
                s_ = (x['truth'] - y['truth']) * order_sign(x, y) if read(x) and read(y) else 0   # D69
                c += s_ > 0; d += s_ < 0; t += s_ == 0
        if c + d + t: U.append((c, d, t))
    return tie_share(U, draws)


def tie_share(U, draws=2000):
    N = sum(sum(u) for u in U)
    if not N: return dict(share=None, ci=None, C=0, D=0, T=0, units=U)
    sh = lambda us: (sum(u[0] for u in us) + 0.5 * sum(u[2] for u in us)) / sum(sum(u) for u in us)
    rng = random.Random(0); bs = sorted(sh(rng.choices(U, k=len(U))) for _ in range(draws))
    return dict(share=sh(U), ci=(bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]), C=sum(u[0] for u in U), D=sum(u[1] for u in U), T=sum(u[2] for u in U), units=U)


def ref_concord(R, MO, p, mode, draws=2000):
    """Order of p across levels under the reference clip: pairs of videos of the same model and
    scene with different prompted values, read under the same reference clip. mode: 'all', 'pure' (both own clips equal to each
    other and to the reference in every other parameter), 'admitted' (pure, and the (scene, p) passed admission.py). interval:
    censored intervals, only separated pairs; best: moving readings with a value. 95% interval over (model, scene) groups."""
    grp = collections.defaultdict(list)
    for r in R:
        if r['param'] != p or r['kind'] not in VAL or not r.get('truth'): continue
        if mode != 'all' and not same_other(r['clip'], r['ref_clip'], p): continue
        if mode == 'admitted' and (not ADM or ADM.get(f"{r['clip'][:3]}|{p}", {}).get('status') != 'pass'): continue
        grp[(r['model'], r['clip'][:3])].append(r)
    per = []
    for rs in grp.values():
        ci = di = cb = db = 0
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                x, y = rs[i], rs[j]
                if x['truth'] == y['truth'] or unseen(x, y, p) or (mode != 'all' and not same_other(x['clip'], y['clip'], p)): continue
                (lx, hx), (ly, hy) = censored(x), censored(y)
                if hx < ly or hy < lx: d = (x['truth'] - y['truth']) * (-1 if hx < ly else 1); ci += d > 0; di += d < 0
                if x.get('best') and y.get('best') and not MO.get((x['key'], p), {}).get('static') and not MO.get((y['key'], p), {}).get('static'):
                    s_ = (x['truth'] - y['truth']) * (x['best'] - y['best']); cb += s_ > 0; db += s_ < 0
        per.append((ci, di, cb, db))
    out = {}
    for nm, k in (('interval', 0), ('best', 2)):
        vals = [(x[k], x[k + 1]) for x in per if x[k] + x[k + 1]]; C, D = sum(v[0] for v in vals), sum(v[1] for v in vals)
        rng = random.Random(0); bs = []
        for _ in range(draws if vals else 0):
            sm = rng.choices(vals, k=len(vals)); c_, d_ = sum(v[0] for v in sm), sum(v[1] for v in sm)
            if c_ + d_: bs.append(c_ / (c_ + d_))
        bs.sort()
        out[nm] = dict(share=C / (C + D) if C + D else None, ci=(bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]) if bs else None, C=C, D=D, groups=len(vals), units=vals)   # units: (C, D) per (model, scene), for pooling groups (main_readme_all.py)
    return out


T = {}; CON = {}; RCON = {}
for label, _, R, MO in G:
    R = [r for r in R if r['clip'] in keep and r['kind'] != 'not_validated' and not (r['param'] == 'mu' and r['clip'][:3] in ('T19', 'T20'))]
    by = collections.defaultdict(list)
    for r in R: by[r['model']].append(r); by['all'].append(r)
    T[label] = {m: stats(v, MO) for m, v in by.items()}
    CON[label] = {m: {p: concord(v, MO, p) for p in ('g', 'mu', 'v')} for m, v in by.items()}
    if label in VIEW:
        VR, VM = VIEW[label]; VR = [r for r in VR if r['clip'] in keep and r['kind'] != 'not_validated']
        vb = collections.defaultdict(list)
        for r in VR: vb[r['model']].append(r); vb['all'].append(r)
        RCON[label] = {m: {p: {md: dict(ref_concord(v, VM, p, md), every_pair=pair_score(v, VM, p, md)) for md in ('admitted', 'pure', 'all')} for p in ('g', 'v')} for m, v in vb.items()}
f = lambda x, d=1: '—' if x is None else f'{x:.{d}f}'
pc = lambda x: '—' if x is None else f'{100 * x:.1f}%'
L = [f'# {a.title}', '', 'Groups:'] + [f'- **{lab}**: `{src}`' for lab, src, _, _ in G]
L += ['', f'Tasks: {len(keep)} ({"present in every group" if a.tasks == "common" else ("all tasks of any group" if a.tasks == "all" else "clips of " + a.tasks.rsplit("/", 1)[-1])}); models: {len(MODELS)}. '
      'Gated results only; mu of T19 / T20 left out (background friction fixed at 1.0).', '']
labels = [g[0] for g in G]
L += ['## Four outcomes (share of parameter results; 95% interval over tasks in brackets)', '', '| model | group | videos | results | did not happen | unexplained | explained | no observation |', '|---|---|---|---|---|---|---|---|']
for m in list(MODELS) + ['all']:
    for lab in labels:
        s = T[lab].get(m)
        if s: L.append(f"| {MODELS.get(m, 'All models')} | {lab} | {s['videos']} | {s['results']} | " + ' | '.join(f"{pc(s[o])} [{100 * s[o + '_ci'][0]:.0f}–{100 * s[o + '_ci'][1]:.0f}]" for o in OUT4) + ' |')
if RCON:
    fs = lambda x: '—' if x['share'] is None else f"{x['share']:.2f} [{x['ci'][0]:.2f}–{x['ci'][1]:.2f}] ({x['C']} / {x['D']})" + (f", tau {2 * x['share'] - 1:+.2f}" if x['share'] is not None else '')
    L += ['', '## Order across levels under the reference clip (main reading)', '',
          'Every video read under its scene\'s reference clip (one library, window and gate per scene). Pairs of videos of the same model and scene '
          'with different prompted values. admitted = pure (own clips equal to each other and to the reference in every other parameter) and the '
          '(scene, parameter) passed the known-truth admission; interval = censored intervals, only separated pairs '
          '(tau = 2 x share - 1); best = moving readings with a value. The null of this view is 0.5 by construction (the same clip for all videos). '
          'mu: see mu_ref.py (only T07, T14 and T18 vary it).', '',
          '| parameter | group | set | interval: share [95%] (C / D) | best: share [95%] (C / D) |', '|---|---|---|---|---|']
    for p in ('g', 'v'):
        for lab in labels:
            if lab not in RCON or 'all' not in RCON[lab]: continue
            for md in ('admitted', 'pure', 'all'):
                x = RCON[lab]['all'][p][md]; L.append(f"| {p} | {lab} | {md} | {fs(x['interval'])} | {fs(x['best'])} |")
    L += ['', '### Per model, admitted set', '', '| model | group | g interval | v interval |', '|---|---|---|---|']
    for m in MODELS:
        for lab in labels:
            x = RCON.get(lab, {}).get(m)
            if x: L.append(f"| {MODELS[m]} | {lab} | {fs(x['g']['admitted']['interval'])} | {fs(x['v']['admitted']['interval'])} |")
    L += ['', 'The table below is the own-clip order (each video under its own clip). It is kept for comparison only: it carries the clip effect.']
L += ['', '## Parameter concordant-pair share (all models pooled; pairs of results in the same scene with different prompted values)', '',
      'Concordant if the read-out (best) follows the prompted order; ties (e.g. both at a library edge) excluded from the share; chance 0.5. '
      'n = results with a prompted value; static = their share that did not move.', '',
      '| parameter | group | concordant share [95% interval] | concordant / discordant | ties | comparable / possible pairs | groups (scene x model) | static | n |', '|---|---|---|---|---|---|---|---|---|']
for p in ('g', 'v'):   # mu: mu_ref.py
    for lab in labels:
        c = CON[lab].get('all', {}).get(p)
        if not c or not c['possible']: continue
        ci = f"[{c['ci'][0]:.2f}–{c['ci'][1]:.2f}]" if c['ci'] else ''
        L.append(f"| {p} | {lab} | {f(c['share'], 2)} {ci} | {c['C']} / {c['D']} | {c['ties']} | {c['comparable']} / {c['possible']} | {c['groups']} | {pc(c['static'])} | {c['n']} |")
L += ['', '## Parameter error (task-balanced median |best / prompt - 1|, %; tasks in brackets) and in-band share', '',
      '| model | group | g (tasks) | mu (tasks) | v (tasks) | prompt in band (n) |', '|---|---|---|---|---|---|']
for m in list(MODELS) + ['all']:
    for lab in labels:
        s = T[lab].get(m)
        if s: L.append(f"| {MODELS.get(m, 'All models')} | {lab} | {f(s['err_g'])} ({s['tasks_g']}) | {f(s['err_mu'])} ({s['tasks_mu']}) | {f(s['err_v'])} ({s['tasks_v']}) | {pc(s['in_band'])} ({s['n_band']}) |")
L += ['', '## Appendix: own-clip concordant-pair share per model (comparison only; carries the clip effect)', '', '| model | group | g: share (C / D, n) | v: share (C / D, n) |', '|---|---|---|---|']
for m in list(MODELS):
    for lab in labels:
        cm = CON[lab].get(m)
        if cm: L.append(f"| {MODELS[m]} | {lab} | " + ' | '.join(f"{f(cm[p]['share'], 2)} ({cm[p]['C']} / {cm[p]['D']}, {cm[p]['n']})" if cm[p]['possible'] else '—' for p in ('g', 'v')) + ' |')
L += ['', '## Appendix: gravity response (median best g of explained point / interval results; n in brackets)', '',
      '| model | group | prompt 1.62 (C01) | prompt 3.71 (C02) | prompt 9.81 (C03) |', '|---|---|---|---|---|']
for m in list(MODELS) + ['all']:
    for lab in labels:
        s = T[lab].get(m)
        if s: L.append(f"| {MODELS.get(m, 'All models')} | {lab} | " + ' | '.join(f"{f(s[f'g_{c}'][0], 2)} ({s[f'g_{c}'][1]})" for c in GP) + ' |')
open(a.out, 'w').write('\n'.join(L) + '\n')
json.dump(dict(stats=T, concordance=CON, reference_order=RCON), open(str(a.out).rsplit('.', 1)[0] + '.json', 'w'), indent=1, default=float)
print(a.out, len(keep), 'tasks;', {lab: T[lab].get('all', {}).get('videos', 0) for lab in labels})
