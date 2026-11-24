"""mu under the main reading: μ changes across clips only in T07, T14 and T18, so every video of these scenes is also inverted for
mu under its scene's reference clip (--ref, default T07_C03, T14_C02, T18_C02: the clip that forms pure pairs with the others where
one exists), as event_ref.py does for g and v; then the concordant-pair share of mu: pairs of videos of the same model, scene and
prompt group with different prompted mu, best values, moving readings; sets 'all' and 'pure' (both clips equal to each other and to
the reference in g and v); the prompt groups pooled by summing pairs, 95 % interval from resampling (model, scene, group) units.
  build WORK              WORK/sam_gen/<SET>: items <key>.ref.<clip> of every finished batch (WORK/results_gen/<batch>_done.txt) of
                          the three scenes whose own clip is not the reference, tracks re-keyed from the batch's tracks_camcomp.jsonl;
                          then evaluate it: eval_generated.py <SET> ... --params mu, motion_check.py, validation_gate.py
  concord TABLES WORK...  rows: <SET>_gated (reference readings) and the own mu rows (evp_all_gated) of every WORK; a video's own row
                          comes from the first WORK that has it; -> TABLES/<OUT>.md / .json
Options (environment): MU_REF JSON {"T07": clip, ...}; MU_SET (default muref); MU_OUT (default MAIN_MU); MU_SPLITS JSON {split: [scenes]}
  for split tables besides 'all'; MU_BATCHES (comma-separated batch names, default: every finished batch evp_b*); MU_OWN (own results,
  default evp_all_gated); MU_EXCLUDE_PAIRS (pairs [key, key, param] left out); MU_EXCLUDE_KEYS (JSON list of keys left out).
Usage: python mu_ref.py build WORK | concord TABLES_DIR WORK [WORK ...]"""
import collections, glob, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from order_rule import order_sign   # D69
REF = {'T07': 'T07_C03', 'T14': 'T14_C02', 'T18': 'T18_C02'}
if os.environ.get('MU_REF'): REF = json.loads(os.environ['MU_REF'])
SET = os.environ.get('MU_SET', 'muref'); OUTN = os.environ.get('MU_OUT', 'MAIN_MU')
VAL = {'point', 'interval', 'upper_bound', 'lower_bound'}
MODELS = {'cogvideox_i2v': 'CogVideoX-1.5', 'cosmos': 'Cosmos-Predict2', 'cosmos3_nano': 'Cosmos3-Nano', 'hunyuan15_i2v': 'HunyuanVideo-1.5',
          'minimax_h3': 'MiniMax-H3', 'physalign_wan22_i2v_a14b': 'PhysAlign', 'physrvg_ti2v5b': 'PhysRVG', 'videogpa_ti2v5b': 'VideoGPA',
          'wan22_i2v_a14b': 'Wan2.2-I2V-A14B', 'wan_ti2v5b': 'Wan2.2-TI2V-5B'}
stage = sys.argv[1]
if stage == 'build':
    W = sys.argv[2]; dst = f'{W}/sam_gen/{SET}'; assert not os.path.exists(dst), dst; os.makedirs(dst)
    RR = f'{W}/reference_reading.json'; READ = json.load(open(RR)) if os.path.exists(RR) else None   # evaluation base: a reading only with a library of the clip's own sample
    from event_ref import reading_clip   # F113 (C3): another clip's library only through the same view; else no second reading
    lib = lambda c: reading_clip(c, 'mu', READ, {(s, 'mu'): r for s, r in REF.items()})
    items, med, n = [], None, 0
    with open(f'{dst}/tracks_camcomp.jsonl', 'w') as o:
        batches = [f'{W}/sam_gen/{b}' for b in os.environ['MU_BATCHES'].split(',')] if os.environ.get('MU_BATCHES') else \
            [f"{W}/sam_gen/{os.path.basename(d)[:-len('_done.txt')]}" for d in sorted(glob.glob(f'{W}/results_gen/evp_b*_done.txt'))]
        for b in batches:
            J = json.load(open(f'{b}/items.json')); med = med or J.get('model_median_map')
            sel = {x['key']: dict(x, key=f"{x['key']}.ref.{lib(x['clip'])}", clip=lib(x['clip']), clip_own=x['clip'], key_own=x['key'], ref_params=['mu'])
                   for x in J['items'] if x.get('map') and x['clip'][:3] in REF and lib(x['clip']) and x['clip'] != lib(x['clip'])}
            if not sel: continue
            items += sel.values()
            for l in open(f'{b}/tracks_camcomp.jsonl'):
                r = json.loads(l); k, _, obj = r['id'].partition('__')
                if k in sel: r['id'] = f"{sel[k]['key']}__{obj}"; o.write(json.dumps(r) + '\n'); n += 1
    json.dump(dict(items=items, model_median_map=med), open(f'{dst}/items.json', 'w'), indent=1)
    print(W, len(items), 'items', n, 'track rows')
elif stage == 'concord':
    T = sys.argv[2]; WORKS = sys.argv[3:]
    SPL = json.load(open(os.environ['MU_SPLITS'])) if os.environ.get('MU_SPLITS') else {}
    XP = {(frozenset(x[:2]), x[2]) for x in json.load(open(os.environ['MU_EXCLUDE_PAIRS']))['clip_pairs']} if os.environ.get('MU_EXCLUDE_PAIRS') else set()
    EXK = set(json.load(open(os.environ['MU_EXCLUDE_KEYS']))) if os.environ.get('MU_EXCLUDE_KEYS') else set()
    own, phys, rows, static = {}, collections.defaultdict(dict), [], {}
    for w in WORKS:
        R = f'{w}/results_gen'
        OWNR = os.environ.get('MU_OWN', 'evp_all_gated')
        for r in map(json.loads, open(f'{R}/{OWNR}.jsonl')):
            if r.get('param') in ('g', 'mu', 'v') and r.get('truth') is not None: phys[r['clip']].setdefault(r['param'], round(r['truth'], 6))
            if r.get('param') == 'mu' and r['clip'][:3] in REF: own.setdefault(r['key'], r)
        for m in map(json.loads, open(f'{R}/{OWNR}_motion.jsonl')):
            if m.get('param') == 'mu': static[m['key']] = bool(m.get('static'))
        IT = {x['key']: x for x in json.load(open(f'{w}/sam_gen/{SET}/items.json'))['items']}
        for m in map(json.loads, open(f'{R}/{SET}_motion.jsonl')):
            if m.get('param') == 'mu' and m['key'] in IT: static[IT[m['key']]['key_own']] = bool(m.get('static'))
        for r in map(json.loads, open(f'{R}/{SET}_gated.jsonl')):
            if r.get('param') != 'mu' or r['key'] not in IT: continue
            o = own.get(IT[r['key']]['key_own'])
            if o: rows.append(dict(r, key=o['key'], clip=o['clip'], truth=o['truth'], model=o['model'], ref_clip=r['clip']))
    from event_ref import cross_view, load_reading   # F113 (C3): a clip whose listed library clip has another view keeps its own mu reading
    READ = load_reading(); REFP = {(s, 'mu'): r for s, r in REF.items()}
    rows += [dict(o, ref_clip=o['clip']) for o in own.values() if o['clip'] == REF[o['clip'][:3]] or cross_view(o['clip'], 'mu', READ, REFP)]
    for m in sorted({r.get('model') for r in rows} - set(MODELS) - {None}): MODELS[m] = m   # a model outside the list keeps its own name
    rows = [r for r in rows if r['key'] not in EXK and r.get('model') in MODELS]
    same_other = lambda x, y: all(phys[x].get(q) == phys[y].get(q) for q in ('g', 'v'))
    unseen = lambda x, y: (frozenset((x['key'], y['key'])), 'mu') in XP

    def units(rs, mode):
        grp = collections.defaultdict(list)
        for r in rs:
            if r['kind'] not in VAL or not r.get('truth') or not r.get('best') or static.get(r['key']): continue
            if mode == 'pure' and not same_other(r['clip'], r['ref_clip']): continue
            grp[(r['model'], r['clip'][:3], r['key'].split('.')[2])].append(r)
        U = []
        for g in grp.values():
            c = d = 0
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    x, y = g[i], g[j]
                    if round(x['truth'], 3) == round(y['truth'], 3) or unseen(x, y) or (mode == 'pure' and not same_other(x['clip'], y['clip'])): continue
                    s = (x['truth'] - y['truth']) * order_sign(x, y); c += s > 0; d += s < 0   # D69
            if c + d: U.append((c, d))
        return U

    def units_tie(rs, mode):   # every pair: concordant 1, discordant 0, no order 0.5 (no moving reading with a value on either side, or equal readings)
        grp = collections.defaultdict(list)
        for r in rs:
            if not r.get('truth'): continue
            if mode == 'pure' and not same_other(r['clip'], r['ref_clip']): continue
            grp[(r['model'], r['clip'][:3], r['key'].split('.')[2])].append(r)
        read = lambda r: r['kind'] in VAL and r.get('best') and not static.get(r['key'])
        U = []
        for g in grp.values():
            c = d = t = 0
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    x, y = g[i], g[j]
                    if round(x['truth'], 3) == round(y['truth'], 3) or unseen(x, y) or (mode == 'pure' and not same_other(x['clip'], y['clip'])): continue
                    s = (x['truth'] - y['truth']) * order_sign(x, y) if read(x) and read(y) else 0   # D69
                    c += s > 0; d += s < 0; t += s == 0
            if c + d + t: U.append((c, d, t))
        return U

    def tie_share(U):
        N = sum(sum(u) for u in U)
        if not N: return None
        sh = lambda us: (sum(u[0] for u in us) + 0.5 * sum(u[2] for u in us)) / sum(sum(u) for u in us)
        rng = random.Random(0); bs = sorted(sh(rng.choices(U, k=len(U))) for _ in range(2000))
        return dict(share=sh(U), ci=(bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]), C=sum(u[0] for u in U), D=sum(u[1] for u in U), T=sum(u[2] for u in U), units=U)

    def share(U):
        C, D = sum(u[0] for u in U), sum(u[1] for u in U)
        if not C + D: return None
        rng = random.Random(0); bs = []
        for _ in range(2000):
            s = rng.choices(U, k=len(U)); c, d = sum(u[0] for u in s), sum(u[1] for u in s)
            if c + d: bs.append(c / (c + d))
        bs.sort(); return dict(share=C / (C + D), ci=(bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]), C=C, D=D, units=U)
    out = {}
    fm = lambda d: '–' if not d else f"{d['share']:.2f} [{d['ci'][0]:.2f}–{d['ci'][1]:.2f}] ({d['C']}/{d['D']})"
    L = ['# μ concordant-pair share, main reading', '', 'μ changes across clips in three scenes only (T07, T14, T18; C04 against the '
         'other clips), so there are few pairs. Reference clips ' + ', '.join(REF[k] for k in sorted(REF)) + '; pure = the two clips equal to each other and to the '
         'reference in g and v. Four prompt groups pooled.', '']
    for split in ['all'] + sorted(SPL):
        sc = set(REF) if split == 'all' else set(SPL[split]) & set(REF)
        L += [f'## {split} ({", ".join(sorted(sc)) or "no scene with μ levels"})', '', '| model | μ, all | μ, pure |', '|---|---|---|']
        for m in list(MODELS) + ['all']:
            rs = [r for r in rows if r['clip'][:3] in sc and (m == 'all' or r['model'] == m)]
            a, p = share(units(rs, 'all')), share(units(rs, 'pure'))
            out[f'{split}|{m}'] = dict(all=a, pure=p, every_pair=tie_share(units_tie(rs, 'all')), every_pair_pure=tie_share(units_tie(rs, 'pure')))
            L.append(f"| {MODELS.get(m, 'All models')} | {fm(a)} | {fm(p)} |")
        L.append('')
    json.dump(out, open(f'{T}/{OUTN}.json', 'w'), indent=1); open(f'{T}/{OUTN}.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
