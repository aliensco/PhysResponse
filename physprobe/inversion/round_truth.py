"""Simulator values per clip for the comparison tables.
  truth_table(work_dirs)   {clip: {param: value}} from the library_index.json of each work dir (the first dir that has a clip wins)
  load(path)               the same table from a JSON file {clip: {param: value}}
  check(PHYS, rows, where) every result row's own value (truth_physical, else truth) must equal PHYS[clip][param]; any difference
                           raises ValueError, so a result can never be paired with the value of another sample of the same clip
Import: from physprobe/inversion, import round_truth"""
import json, os


def truth_table(work_dirs):
    out = {}
    for w in work_dirs:
        f = os.path.join(w, 'library_index.json')
        if not os.path.exists(f): continue
        here = {}
        for L in json.load(open(f))['libraries']:
            for c, t in zip(L['members'], L['truths']): here.setdefault(c, {}).setdefault(L['param'], round(t, 6))
        for c, d in here.items(): out.setdefault(c, d)   # a clip takes every value from the first dir that has it
    return out


def load(path):
    return {c: {p: round(v, 6) for p, v in d.items()} for c, d in json.load(open(path)).items()}


def check(PHYS, rows, where):
    bad = []
    for r in rows:
        p = r.get('param'); v = r.get('truth_physical', r.get('truth'))
        if p not in ('g', 'mu', 'v') or v is None or r.get('clip') not in PHYS or p not in PHYS[r['clip']]: continue
        if abs(round(v, 6) - PHYS[r['clip']][p]) > 1e-6 * max(1, abs(v)): bad.append((r.get('key'), r['clip'], p, v, PHYS[r['clip']][p]))
    if bad: raise ValueError(f'{where}: {len(bad)} result rows whose value differs from the per-clip table, e.g. {bad[:5]}')
