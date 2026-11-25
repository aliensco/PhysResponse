"""Generated videos of the four-prompt evaluation -> one batch of SAM 3.1 jobs and items (the make_gen_jobs.py format).
Input: a folder of manifest shards (*.jsonl), one row per generated video with id, model, clip_id, prompt_group (short, standard,
detailed or concat), seed, video (resized to the reference resolution) with video_sha256, condition_image with condition_sha256, and
optionally condition_frame_index (the reference frame the condition image shows, default 0), fps, frames and the prompt fields.
Checks per row: known clip with a predictability window (or listed in --whole-video-clips), the selected models, prompt groups and
seed, both files present with the stated sha256, and the same condition image as every earlier row of the clip. Rows that fail are
listed in PHYSPROBE_WORK/rejected.jsonl with the reasons; rows already in an earlier batch are skipped. Item key: model.clip.group.
The registration fallback (a model's median mapping) can be taken from an earlier items.json (--median-from).
Usage (from physprobe/inversion, PHYSPROBE_WORK set): python event_items.py NAME MANIFEST_DIR [--median-from ITEMS.json]
  [--models m1,m2] [--groups short,standard,detailed,concat] [--seed 0] [--max 24] [--whole-video-clips C1,C2] [--workers 8]
  -> PHYSPROBE_WORK/sam_gen/NAME/jobs.jsonl and items.json; exit 3 when no new row is taken"""
import argparse, glob, hashlib, json, os, sys
from pathlib import Path
sys.path.insert(0, os.getcwd())
import common as c2
import make_gen_jobs as mg

ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('manifest_dir'); ap.add_argument('--median-from')
ap.add_argument('--models', help='comma-separated model names (default: every model of the manifest)')
ap.add_argument('--groups', default='short,standard,detailed,concat'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--max', type=int, default=24, help='rows per batch'); ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--whole-video-clips', default='', help='clips without a predictability window that are still tracked over the whole video')
a = ap.parse_args()
MODELS = set(a.models.split(',')) if a.models else None
GROUPS = set(a.groups.split(',')); ALLOW = set(a.whole_video_clips.split(',')) - {''}
done = set()
for f in glob.glob(str(c2.B / 'sam_gen' / '*' / 'items.json')): done |= {x.get('manifest_id', x['key']) for x in json.load(open(f))['items']}
REJ = c2.B / 'rejected.jsonl'
rejected = {json.loads(l)['id'] for l in open(REJ)} if REJ.exists() else set()
rows = {}
for f in sorted(glob.glob(os.path.join(a.manifest_dir, '*.jsonl'))):
    for l in open(f):
        if not l.strip(): continue
        m = json.loads(l)
        if m.get('prompt_group') == 'final': m = dict(m, prompt_group=m.get('annotation'))   # released manifests: group 'final', level in 'annotation'
        rows.setdefault(m['id'], m)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
CS = c2.B / 'condition_sha.json'   # clip -> condition sha256 of the first accepted row; every row of a clip must share it
cond = json.load(open(CS)) if CS.exists() else {}
take, rej = [], []
for k, m in sorted(rows.items()):
    if k in done or k in rejected: continue
    why = []
    if MODELS is not None and m.get('model') not in MODELS: why.append(f"model {m.get('model')} not selected")
    if m.get('prompt_group') not in GROUPS: why.append(f"prompt_group {m.get('prompt_group')}")
    if m.get('seed') != a.seed: why.append(f"seed {m.get('seed')}")
    if m.get('clip_id') not in mg.RD: why.append(f"clip {m.get('clip_id')} unknown")
    elif mg.HZ.get(m['clip_id'], {}).get('status') != 'OK' and m['clip_id'] not in ALLOW: why.append(f"clip {m['clip_id']} has no predictability window")
    for f_, h_ in (('video', 'video_sha256'), ('condition_image', 'condition_sha256')):
        if not m.get(f_) or not Path(m[f_]).exists(): why.append(f'{f_} missing')
        elif sha(m[f_]) != m.get(h_): why.append(f'{f_} sha256 differs from {h_}')
    if not why and cond.setdefault(m['clip_id'], m['condition_sha256']) != m['condition_sha256']: why.append('condition image differs from the earlier rows of this clip')
    (rej if why else take).append((m, why))
if rej:
    with open(REJ, 'a') as o:
        for m, why in rej: o.write(json.dumps(dict(id=m['id'], why=why)) + '\n')
    print(len(rej), 'rows rejected (rejected.jsonl):', [(m['id'], w) for m, w in rej[:5]])
json.dump(cond, open(CS, 'w'), indent=1)
take = [m for m, _ in take][:a.max]
if not take: print('no new items'); sys.exit(3)
CI = c2.B / 'sam_gen' / a.name / 'conditions'; CI.mkdir(parents=True, exist_ok=True); mg.CI = CI
KEY = lambda m: f"{m['model']}.{m['clip_id']}.{m['prompt_group']}"   # the tracker joins video and object ids with '__', so keys use dots
items = []
for m in take:
    if not (CI / f"{KEY(m)}.png").is_symlink(): (CI / f"{KEY(m)}.png").symlink_to(m['condition_image'])
    items.append(dict(key=KEY(m), manifest_id=m['id'], seed=m['seed'], annotation=0, model=m['model'], clip=m['clip_id'], item=KEY(m),
                      path=m['video'], experiment='EVP', prompt_group=m['prompt_group'], prompt_source=m.get('prompt_source'),
                      prompt_version=m.get('prompt_version'), prompt_sha256=m.get('prompt_sha256'), condition_sha256=m['condition_sha256'],
                      video_sha256=m['video_sha256'], text_tokens=m.get('text_tokens'), text_limit=m.get('text_limit'),
                      text_truncated=m.get('text_truncated'), manifest_frames=m.get('frames'), manifest_fps=m.get('fps'), image=m.get('image'),
                      condition_frame_index=int(m.get('condition_frame_index', 0))))
mg.build(items, a.name, str(c2.B / 'sam_gen' / a.name), a.workers, a.median_from)
d = json.load(open(c2.B / 'sam_gen' / a.name / 'items.json'))
odd = [(x['key'], x['fps'], x['frames'], x['manifest_fps'], x['manifest_frames']) for x in d['items']
       if x['manifest_fps'] and (abs(x['fps'] - x['manifest_fps']) > 0.01 or x['frames'] != x['manifest_frames'])]
if odd: print('fps / frames differ from the manifest:', odd[:5])
