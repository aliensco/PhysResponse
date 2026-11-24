"""Where code and data live. Code: this directory (in git). Data (simulation libraries, caches, tracks, results; never in git):
PHYSPROBE_WORK. The released evaluation base is one self-contained folder: besides the libraries, windows, box cache, known-answer
results and known-truth sets it holds ready/ (reference bindings, paths relative to the base), projection/cams.json,
known_conditions/ (readiness of the 100 tasks, T13 metadata, the prompts' declared friction) and bindings/; when PHYSPROBE_WORK has
a ready/ folder every path below is taken from it. Otherwise (the research layout) PHYSPROBE_ROOT is the root of the project tree:
  .local-research/evaluation/core100_inversion_workspace/{evaluation,projection,ready}, .local-research/evaluation/known_conditions
gt() maps the absolute local paths recorded in data files into PHYSPROBE_ROOT, and paths relative to the base into PHYSPROBE_WORK."""
import os, re
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = os.environ.get('PHYSPROBE_ROOT', str(Path.cwd()))
LOCAL_ROOTS = tuple(filter(None, os.environ.get('PHYSPROBE_LEGACY_ROOTS', '').split(os.pathsep)))
EVALUATION = Path(ROOT) / '.local-research/evaluation'
WORKSPACE = EVALUATION / 'core100_inversion_workspace'
WORK = Path(os.environ.get('PHYSPROBE_WORK', WORKSPACE / 'evaluation'))
BASE = (WORK / 'ready').exists()   # self-contained evaluation base
PROJECTION = WORK / 'projection' if BASE else WORKSPACE / 'projection'
READY = WORK / 'ready' if BASE else WORKSPACE / 'ready'
KNOWN = WORK / 'known_conditions' if BASE else EVALUATION / 'known_conditions'
PREPARATION = EVALUATION / 'preparation'
GENERATED = Path(ROOT) / 'data/generated'
READINESS = KNOWN / 'readiness100.jsonl'   # one row per task (100 tasks)


def chunk_name(tag):
    """local file name of a library chunk: the job tag <task>_<param>_<chunk> (tags of older plans carry one prefix word);
    other tags (host checks, horizon runs) are kept as they are"""
    m = re.match(r'[A-Za-z0-9]+_(T\d\d_C\d\d_(?:g|mu|v)_[A-Za-z0-9]+)$', tag)
    return m.group(1) if m else tag


def gt(p):
    """map a recorded absolute local path into ROOT (identity when running locally); a relative path is relative to the base"""
    s = str(p)
    if BASE and not os.path.isabs(s): return str(WORK / s)
    for lr in LOCAL_ROOTS:
        if s.startswith(lr): return ROOT + s[len(lr):]
    return s
