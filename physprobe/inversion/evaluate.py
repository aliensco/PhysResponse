"""Evaluate one generated model on Core100 with one command.

    python -m physprobe.inversion.evaluate --videos MANIFEST_DIR --name MODEL [--from STEP | --only STEP] [--gpu 0] [--workers 16]

MANIFEST_DIR holds the generation manifest shards (*.jsonl, one row per video: id, model, clip_id, prompt_group, seed, video,
video_sha256, condition_image, condition_sha256). The base workspace PHYSPROBE_WORK holds the Core100 libraries, windows, known-answer
results and known-truth sets; every file of this run goes to PHYSPROBE_WORK/NAME/, a workspace that links the base read-only.
Steps, in order. Without --from / --only a step is skipped when its output already exists, so an interrupted run resumes where it
stopped; --from STEP runs STEP and every later step again; the first failing step stops the run and names its log:

  workspace  PHYSPROBE_WORK/NAME with links to the base workspace and its known-truth sets
  items      event_items.py: condition-image registration and SAM prompts of every video
  audit      prompt_audit.py: compared pairs the model could not tell apart (same video, or the difference cut by its text limit)
  track      server/sam_tracker.py: SAM 3.1 tracks of every object (GPU)
  camera     cam_comp.py: camera compensation
  invert     eval_generated.py: g, mu and v against each video's own task
  motion     motion_check.py: did the prompted motion happen
  static     static_rule.py: clips whose reference does not move are scored by whether the video stays still
  reference  event_ref.py, eval_generated.py, motion_check.py: g and v under the scene's reference clip
  gate       validation_gate.py: keep the results whose task and format pass the known-truth test
  friction   apply_declared_mu.py: mu scored against the friction each prompt declares
  view       event_view.py: every g / v result read under its scene's reference clip
  compare    event_compare.py: per prompt group and pooled
  mu         mu_ref.py: mu under the reference clips of the scenes where it varies
  visual     common_metrics.py: the V metrics
  table      main_readme_all.py: the main-table row of the model -> PHYSPROBE_WORK/NAME/tables/MAIN_README_ALL.md / .json
"""
import argparse, glob, json, os, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ['workspace', 'items', 'audit', 'track', 'camera', 'invert', 'motion', 'static', 'reference', 'gate', 'friction', 'view', 'compare', 'mu', 'visual', 'kinematics', 'table']
GROUPS = ('short', 'medium', 'detailed', 'concat')


def args():
    ap = argparse.ArgumentParser(prog='python -m physprobe.inversion.evaluate', description='Evaluate one generated model on Core100 (steps: ' + ', '.join(STEPS) + ')')
    ap.add_argument('--videos', required=True, help='folder of generation manifest shards (*.jsonl)')
    ap.add_argument('--name', required=True, help='run name; files go to PHYSPROBE_WORK/NAME/')
    ap.add_argument('--from', dest='start', choices=STEPS, help='run this step and every later one again (earlier outputs are kept)')
    ap.add_argument('--only', choices=STEPS, help='run this step only')
    ap.add_argument('--gpu', default='0', help='GPU for tracking (CUDA_VISIBLE_DEVICES)')
    ap.add_argument('--workers', type=int, default=16, help='parallel processes of the CPU steps')
    ap.add_argument('--reference-clips', help='JSON {"<scene>|<param>": clip} (default: BASE/reference_clips.json)')
    ap.add_argument('--known-conditions', help='folder with known_conditions_<group>.json of the prompt release (default: BASE/known_conditions)')
    ap.add_argument('--sam3', default=os.environ.get('SAM3_SOURCE'), help='SAM 3.1 source tree (default: $SAM3_SOURCE)')
    ap.add_argument('--checkpoint', default=os.environ.get('SAM3_CHECKPOINT'), help='sam3.1_multiplex.pt (default: $SAM3_CHECKPOINT)')
    ap.add_argument('--ref-videos', help='folder of the reference videos <clip>.mp4 the V metrics compare with (default: BASE/reference_videos)')
    ap.add_argument('--tokenizer', help="the model's text tokenizer (Hugging Face folder) for the prompt audit; without it no prompt is cut")
    ap.add_argument('--token-limit', type=int, help="the model's text limit in tokens (e.g. 512 for umT5, 226 for CogVideoX)")
    ap.add_argument('--no-special-tokens', action='store_true', help='count the prompt tokens without special tokens (e.g. HunyuanVideo-1.5)')
    ap.add_argument('--median-from', help='items.json of an earlier run whose registrations join the per-model fallback mapping')
    ap.add_argument('--python', default=sys.executable, help='interpreter of the steps (default: this one)')
    return ap.parse_args()


def main():
    a = args()
    base = Path(os.environ.get('PHYSPROBE_WORK') or sys.exit('set PHYSPROBE_WORK to the base workspace')).resolve()
    W = base / a.name; S, R = W / 'sam_gen', W / 'results_gen'; T = W / 'tables'; N = a.name
    refc = a.reference_clips or str(base / 'reference_clips.json'); kc = Path(a.known_conditions or base / 'known_conditions')
    refv = a.ref_videos or (str(base / 'reference_videos') if (base / 'reference_videos').is_dir() else None)
    adm = ['--admission', str(base / 'reference_admission.json')] if (base / 'reference_admission.json').exists() else []
    A = W / 'audit'; tx = ['--exclude-keys', A / 'trunc_group_keys.json', '--exclude-pairs', A / 'trunc_exclusions.json']
    quarter = (base / 'QUARTER_ON').exists()   # bases whose libraries carry quarter steps (libraries_quarter, box_cache_quarter, gate tag quarter)
    env = dict(os.environ, PHYSPROBE_WORK=str(W), OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PHYSPROBE_MOTION_WINDOW='fixed', CUDA_VISIBLE_DEVICES='')
    for k in ('PHYSPROBE_EXTRA_LIBS', 'PHYSPROBE_BOXCACHE', 'PHYSPROBE_SHIFT_RULE', 'PHYSPROBE_REF_OUT', 'PHYSPROBE_KNOWN_TAG'): env.pop(k, None)
    if quarter: env.update(PHYSPROBE_EXTRA_LIBS='libraries_quarter', PHYSPROBE_BOXCACHE='box_cache_quarter')
    gt = ['quarter'] if quarter else []
    py = a.python

    def run(step, *cmd, gpu=False, ok=(0,), extra_env=None):
        e = dict(env, **(extra_env or {}))
        if gpu: e['CUDA_VISIBLE_DEVICES'] = a.gpu
        (W / 'logs').mkdir(parents=True, exist_ok=True); log = W / 'logs' / f'{step}.log'
        with open(log, 'a') as f:
            f.write(f"$ {' '.join(map(str, cmd))}\n"); f.flush()
            rc = subprocess.call([str(c) for c in cmd], cwd=HERE, env=e, stdout=f, stderr=subprocess.STDOUT)
        if rc not in ok: sys.exit(f'step {step} failed (exit {rc}); log: {log}')
        return rc

    def has(*paths): return all(Path(p).exists() and Path(p).stat().st_size > 0 for p in paths)

    def lines(p): return sum(1 for l in open(p) if l.strip()) if Path(p).exists() else 0

    def step_workspace():
        S.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True)
        for f in base.iterdir():
            if f.name in ('sam_gen', 'results_gen', 'logs', 'rejected.jsonl', 'condition_sha.json', 'progress.log') or f.resolve() == W: continue
            if not (W / f.name).exists() and not (W / f.name).is_symlink(): (W / f.name).symlink_to(f.resolve())
        for pat in ('s*.jsonl', 'common_synth_*.jsonl', 'common_frozen_f0.jsonl', 'kin_synth_*.jsonl'):   # known-truth sets of the gate, reference rows of the table
            for f in (base / 'results_gen').glob(pat):
                if not (R / f.name).exists(): (R / f.name).symlink_to(f.resolve())
        (W / 'WORKSPACE_OK').write_text(str(base) + '\n')

    def step_items():
        cmd = [py, 'event_items.py', N, a.videos, '--max', '1000000', '--workers', a.workers]
        if a.median_from: cmd += ['--median-from', a.median_from]
        if (base / 'static_rule.json').exists():   # clips whose reference does not move have no predictability window: tracked over the whole video, scored by the static step
            cmd += ['--whole-video-clips', ','.join(json.load(open(base / 'static_rule.json'))['clips'])]
        run('items', *cmd)
        V = S / N / 'videos'; V.mkdir(exist_ok=True)
        for l in open(S / N / 'upload.txt'):
            src, dst = l.rstrip('\n').split('\t')
            if not (V / dst).exists(): (V / dst).symlink_to(src)

    def step_audit():
        cmd = [py, 'prompt_audit.py', a.videos, A]
        if a.tokenizer: cmd += ['--tokenizer', a.tokenizer] + (['--token-limit', a.token_limit] if a.token_limit else []) + (['--no-special-tokens'] if a.no_special_tokens else [])
        run('audit', *cmd)

    def step_track():
        if not (a.sam3 and a.checkpoint): sys.exit('step track needs --sam3 and --checkpoint (or $SAM3_SOURCE / $SAM3_CHECKPOINT)')
        run('track', py, 'server/sam_tracker.py', S / N / 'jobs.jsonl', S / N / 'tracks_lean.jsonl', '--sam3', a.sam3, '--checkpoint', a.checkpoint, gpu=True)

    def step_camera():
        run('camera', py, 'cam_comp.py', S / N / 'items.json', S / N / 'tracks_lean.jsonl', '--out', S / N / 'tracks_camcomp.jsonl', '--workers', a.workers)

    def step_invert(): run('invert', py, 'eval_generated.py', N, S / N / 'tracks_camcomp.jsonl', '--clock', 'realtime', '--workers', a.workers)

    def step_motion(): run('motion', py, 'motion_check.py', N, S / N / 'tracks_camcomp.jsonl', '--items', N, '--workers', a.workers)

    def step_static():
        if (base / 'static_rule.json').exists(): run('static', py, 'static_rule.py', N)
        else: (R / f'{N}_static.json').write_text('{}\n')

    def step_reference():
        if run('reference', py, 'event_ref.py', N, refc, ok=(0, 3)) == 3: (R / f'{N}_reference.done').write_text('none needed\n'); return
        run('reference', py, 'eval_generated.py', f'{N}_ref', S / f'{N}_ref' / 'tracks_camcomp.jsonl', '--params', 'g,v', '--clock', 'realtime', '--workers', a.workers)
        run('reference', py, 'motion_check.py', f'{N}_ref', S / f'{N}_ref' / 'tracks_camcomp.jsonl', '--items', f'{N}_ref', '--workers', a.workers)
        (R / f'{N}_reference.done').write_text('ok\n')

    def step_gate():
        run('gate', py, 'validation_gate.py', N, N, *gt)
        if has(R / f'{N}_ref.jsonl'): run('gate', py, 'validation_gate.py', f'{N}_ref', f'{N}_ref', *gt)
        for x in (N, f'{N}_ref'):   # the gate keeps every row; the motion records of the gated set are those of the set
            if has(R / f'{x}_motion.jsonl') and not (R / f'{x}_gated_motion.jsonl').exists(): (R / f'{x}_gated_motion.jsonl').symlink_to(f'{x}_motion.jsonl')

    def step_friction():
        for g in GROUPS:
            k = kc / f'known_conditions_{g}.json'
            if k.exists(): run('friction', py, 'apply_declared_mu.py', f'{N}_gated', k, '--prompts', g)
        (R / f'{N}_friction.done').write_text('ok\n')

    def step_view():
        if not has(R / f'{N}_ref_gated.jsonl'):   # nothing needed a second reading: the view is the own-clip result
            shutil.copy(R / f'{N}_gated.jsonl', R / f'{N}_view_gated.jsonl'); shutil.copy(R / f'{N}_gated_motion.jsonl', R / f'{N}_view_gated_motion.jsonl'); return
        run('view', py, 'event_view.py', W, f'{N}_gated', f'{N}_ref_gated', f'{N}_ref', refc, f'{N}_view_gated')

    def step_compare():
        T.mkdir(exist_ok=True)
        G = [f'{g}={N}_gated:*.{g}@{N}_view_gated' for g in GROUPS]
        run('compare', py, 'event_compare.py', W, T / 'groups.md', *G, '--tasks', 'all', '--models', ','.join(models()), *adm, *tx, '--title', f'{N}: per prompt group')
        run('compare', py, 'event_compare.py', W, T / 'pooled.md', f'all={N}_gated@{N}_view_gated', '--tasks', 'all', '--models', ','.join(models()), *adm, *tx, '--title', f'{N}: prompt groups pooled')

    def models(): return sorted({x['model'] for x in json.load(open(S / N / 'items.json'))['items']})

    def step_mu():
        me = dict(MU_SET=f'{N}_muref', MU_OUT='MAIN_MU', MU_BATCHES=N, MU_OWN=f'{N}_gated', MU_EXCLUDE_PAIRS=str(A / 'trunc_exclusions.json'), MU_EXCLUDE_KEYS=str(A / 'trunc_group_keys.json'))
        if not (S / f'{N}_muref').exists(): run('mu', py, 'mu_ref.py', 'build', W, extra_env=me)
        if lines(S / f'{N}_muref' / 'tracks_camcomp.jsonl'):
            run('mu', py, 'eval_generated.py', f'{N}_muref', S / f'{N}_muref' / 'tracks_camcomp.jsonl', '--params', 'mu', '--clock', 'realtime', '--workers', a.workers)
            run('mu', py, 'motion_check.py', f'{N}_muref', S / f'{N}_muref' / 'tracks_camcomp.jsonl', '--items', f'{N}_muref', '--workers', a.workers)
            run('mu', py, 'validation_gate.py', f'{N}_muref', f'{N}_muref', *gt)
        else:
            for f in (f'{N}_muref_gated.jsonl', f'{N}_muref_motion.jsonl'): (R / f).write_text('')
        T.mkdir(exist_ok=True); run('mu', py, 'mu_ref.py', 'concord', T, W, extra_env=me)

    def step_visual():
        run('visual', py, 'common_metrics.py', N, S / N / 'tracks_camcomp.jsonl', '--out', N, '--clock', 'realtime', '--workers', a.workers,
            *(['--ref-videos', refv] if refv else []))

    def step_kinematics():   # displacement along the simulated direction, causal order, acceleration exponent (2026-11-10)
        run('kinematics', py, 'kinematics.py', f'{N}_gated', N, S / N / 'tracks_camcomp.jsonl', R / f'kin_{N}.jsonl', 'realtime', a.workers)

    def step_table():
        run('table', py, 'main_readme_all.py', T, W, '--groups', T / 'groups.json', '--pooled', T / 'pooled.json', '--mu', T / 'MAIN_MU.json',
            '--common', R / f'common_{N}.jsonl', '--results', R / f'{N}_gated.jsonl', '--view', R / f'{N}_view_gated.jsonl',
            '--mu-readings', f'{W}:{N}_muref', *(['--kin', R / f'kin_{N}.jsonl'] if has(R / f'kin_{N}.jsonl') else []),
            *(['--exclusions', A / 'trunc_exclusions.json'] if has(A / 'trunc_exclusions.json') else []))   # F110 A2
        d = json.load(open(T / 'MAIN_README_ALL.json'))
        for m in models():
            o = d.get(m, {}); nf = lambda k, f: '–' if o.get(k) is None else format(o[k], f)
            c = lambda p: '–' if (((o.get('concordant') or {}).get(p) or {}).get('normalized')) is None else f"{o['concordant'][p]['normalized']:+.2f}"
            e = lambda p: '–' if ((o.get('coverage') or {}).get(p) or {}).get('err') is None else f"{100 * o['coverage'][p]['err']:.1f}%"
            print(f"{m}: concordance g {c('g')} v {c('v')} mu {c('mu')} | error g {e('g')} mu {e('mu')} v {e('v')} | "
                  f"L2 {nf('l2', '.2f')}% | S-IoU {nf('s_iou', '.3f')} | PSNR {nf('psnr', '.1f')}")
        print(T / 'MAIN_README_ALL.md')

    done = {'workspace': lambda: has(W / 'WORKSPACE_OK'), 'items': lambda: has(S / N / 'items.json', S / N / 'jobs.jsonl'), 'audit': lambda: has(A / 'trunc_exclusions.json'),
            'track': lambda: has(S / N / 'tracks_lean.jsonl') and lines(S / N / 'tracks_lean.jsonl') >= sum(len(json.loads(l)['objects']) for l in open(S / N / 'jobs.jsonl')),
            'camera': lambda: has(S / N / 'tracks_camcomp.jsonl'), 'invert': lambda: has(R / f'{N}.jsonl'), 'motion': lambda: has(R / f'{N}_motion.jsonl'), 'static': lambda: has(R / f'{N}_static.json'),
            'reference': lambda: has(R / f'{N}_reference.done'),
            'gate': lambda: has(R / f'{N}_gated.jsonl'), 'friction': lambda: has(R / f'{N}_friction.done'), 'view': lambda: has(R / f'{N}_view_gated.jsonl'),
            'compare': lambda: has(T / 'groups.json', T / 'pooled.json'), 'mu': lambda: has(T / 'MAIN_MU.json'),
            'visual': lambda: has(R / f'common_{N}.jsonl'), 'kinematics': lambda: has(R / f'kin_{N}.jsonl'), 'table': lambda: False}
    loc = dict(locals()); fn = {k: v for k, v in loc.items() if k.startswith('step_')}
    todo = [a.only] if a.only else STEPS[STEPS.index(a.start):] if a.start else STEPS
    for s in todo:
        if s != 'table' and not (a.only or a.start) and done[s](): print(f'{s}: done, skipped'); continue   # --from / --only: these steps run again
        print(f'{s} ...', flush=True); fn[f'step_{s}']()
        if s != 'table' and not done[s](): sys.exit(f'step {s} finished without its output; log: {W / "logs"}')


if __name__ == '__main__':
    main()
