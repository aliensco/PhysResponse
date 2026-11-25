"""Prompt audit of one model: which compared pairs the model could not tell apart, so that they leave the concordant-pair shares.
Cross-clip pairs (same group and scene, two clips whose prompted value of a parameter differs): 'unseen' if both videos are the
same file, if the text the model sees is identical, or if a span where the two prompts differ and that carries the prompted value
lies beyond the model's text limit on either side. Cross-group pairs (same clip, two of short / standard / detailed): 'unseen' if
the videos are the same file or the visible texts are identical. The visible text is the prompt cut after the last token kept by
the model's tokenizer (--tokenizer, a Hugging Face tokenizer folder, and --token-limit); without a tokenizer no text is cut.
Prompted values: the library truths of the base (library_index.json).
Usage (from physprobe/inversion, PHYSPROBE_WORK set): python prompt_audit.py MANIFEST_DIR OUT_DIR [--tokenizer DIR --token-limit N
  [--no-special-tokens]]  -> OUT_DIR/trunc_exclusions.json (clip_pairs [key, key, param], group_pairs [key, key]), trunc_pairs.json"""
import argparse, collections, difflib, glob, itertools, json, os, re

from paths import WORK as B

ap = argparse.ArgumentParser(); ap.add_argument('manifest_dir'); ap.add_argument('out')
ap.add_argument('--tokenizer'); ap.add_argument('--token-limit', type=int); ap.add_argument('--no-special-tokens', action='store_true')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
tok = None
if a.tokenizer:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, use_fast=True)

rows = {}
for f in sorted(glob.glob(os.path.join(a.manifest_dir, '*.jsonl'))):
    for l in open(f):
        if not l.strip(): continue
        m = json.loads(l)
        g = m.get('annotation') if m.get('prompt_group') == 'final' else m.get('prompt_group')   # as event_items.py: group 'final', level in 'annotation'
        key = f"{m['model']}.{m['clip_id']}.{g}"
        text = m['prompt'] if m.get('prompt') is not None else open(m['prompt_file']).read()
        rows[key] = dict(key=key, model=m['model'], clip=m['clip_id'], group=g, text=text, sha=m.get('video_sha256'))
for r in rows.values():
    t = r['text']; r.update(tokens=None, truncated=0, cut=len(t))
    if tok is not None:
        enc = tok(t, return_offsets_mapping=True, add_special_tokens=not a.no_special_tokens); n = len(enc['input_ids'])
        if a.token_limit and n > a.token_limit:
            offs = [o for o in enc['offset_mapping'][:a.token_limit] if o[1] > o[0]]; r['cut'] = offs[-1][1] if offs else 0
        r.update(tokens=n, truncated=max(0, n - a.token_limit) if a.token_limit else 0)

PT = {}
for L in json.load(open(B / 'library_index.json'))['libraries']:
    for c, t in zip(L['members'], L['truths']): PT.setdefault((c, L['param']), t)
WORD = re.compile(r'\S+'); NUM = re.compile(r'-?\d+(?:\.\d+)?')


def spans(x, y):
    """differing spans as ((x0, x1), (y0, y1)) character ranges, word diff"""
    wa, wb = [(m.start(), m.end(), m.group()) for m in WORD.finditer(x)], [(m.start(), m.end(), m.group()) for m in WORD.finditer(y)]
    sm = difflib.SequenceMatcher(None, [w[2] for w in wa], [w[2] for w in wb], autojunk=False); res = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal': continue
        ra = (wa[i1][0], wa[i2 - 1][1]) if i2 > i1 else (wa[i1][0] if i1 < len(wa) else len(x),) * 2
        rb = (wb[j1][0], wb[j2 - 1][1]) if j2 > j1 else (wb[j1][0] if j1 < len(wb) else len(y),) * 2
        res.append((ra, rb))
    return res


def carries(s, val):
    return val and any(abs(float(n) / val - 1) < 5e-3 or abs(float(n) / (val * 100) - 1) < 5e-3 for n in NUM.findall(s) if float(n))


unseen, excl_clip, excl_grp = [], [], []
by = collections.defaultdict(list)
for r in rows.values(): by[(r['model'], r['group'], r['clip'][:3])].append(r)
for (m, g, task), rs in by.items():
    for x, y in itertools.combinations(sorted(rs, key=lambda r: r['clip']), 2):
        vis_same = x['text'][:x['cut']] == y['text'][:y['cut']]; same_video = x['sha'] and x['sha'] == y['sha']; sp = None
        for p in ('g', 'mu', 'v'):
            tx, ty = PT.get((x['clip'], p)), PT.get((y['clip'], p))
            if tx is None or ty is None or round(tx, 6) == round(ty, 6): continue
            why = 'identical video' if same_video else 'identical visible text' if vis_same else None; located = True
            if not why:
                sp = sp if sp is not None else spans(x['text'], y['text'])
                rel = [s for s in sp if carries(x['text'][s[0][0]:s[0][1]], tx) or carries(y['text'][s[1][0]:s[1][1]], ty)]
                if not rel: rel, located = sp, False
                if any(s[0][1] > x['cut'] or s[1][1] > y['cut'] for s in rel): why = 'value cut'
            if why:
                unseen.append(dict(kind='clip', model=m, group=g, clips=[x['clip'], y['clip']], param=p, reason=why, located=located))
                excl_clip.append([x['key'], y['key'], p])
by = collections.defaultdict(dict)
for r in rows.values():
    if r['group'] in ('short', 'standard', 'detailed'): by[(r['model'], r['clip'])][r['group']] = r
for (m, c), d in by.items():
    for g1, g2 in (('short', 'standard'), ('standard', 'detailed'), ('short', 'detailed')):
        if g1 not in d or g2 not in d: continue
        x, y = d[g1], d[g2]
        why = 'identical video' if x['sha'] and x['sha'] == y['sha'] else 'identical visible text' if x['text'][:x['cut']] == y['text'][:y['cut']] else None
        if why: unseen.append(dict(kind='group', model=m, clip=c, groups=[g1, g2], reason=why)); excl_grp.append([x['key'], y['key']])
json.dump(unseen, open(os.path.join(a.out, 'trunc_pairs.json'), 'w'), indent=1)
json.dump(dict(clip_pairs=excl_clip, group_pairs=excl_grp), open(os.path.join(a.out, 'trunc_exclusions.json'), 'w'))
json.dump(sorted({k for x in excl_grp for k in x}), open(os.path.join(a.out, 'trunc_group_keys.json'), 'w'))
print(len(rows), 'prompts,', sum(r['truncated'] > 0 for r in rows.values()), 'cut by the text limit;', len(excl_clip), 'cross-clip and',
      len(excl_grp), 'cross-group pairs left out', dict(collections.Counter(u['reason'] for u in unseen)))
