"""Manifest of one model's videos for evaluate.py. The base lists the 400 generation inputs (generation_inputs.jsonl: task, prompt
group, prompt, condition image, reference size); VIDEO_DIR holds one video per input, named <clip_id>.<prompt_group>.mp4, generated
with seed 0 from that condition image and prompt. Each video is resized to the reference size of its task with ffmpeg (Lanczos,
frame rate kept, H.264), as the evaluated models' videos were, into OUT_DIR/videos/, and listed in OUT_DIR/manifest.jsonl with the
sha256 of the video and of its condition image. Missing videos are listed and left out.
Usage (PHYSPROBE_WORK set): python -m physprobe.inversion.make_manifest --videos VIDEO_DIR --model NAME --out OUT_DIR [--ffmpeg PATH]"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser(prog='python -m physprobe.inversion.make_manifest', description=__doc__.split('\n')[0])
ap.add_argument('--videos', required=True, help='folder with <clip_id>.<prompt_group>.mp4'); ap.add_argument('--model', required=True, help='model name of the rows')
ap.add_argument('--out', required=True, help='folder of the manifest and the resized videos'); ap.add_argument('--ffmpeg', help='ffmpeg binary (default: ffmpeg on PATH, else imageio-ffmpeg)')
a = ap.parse_args()
B = Path(os.environ.get('PHYSPROBE_WORK') or sys.exit('set PHYSPROBE_WORK to the evaluation base')).resolve()
ff = a.ffmpeg or shutil.which('ffmpeg')
if not ff:
    try: import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError: sys.exit('no ffmpeg: install it or pip install imageio-ffmpeg, or pass --ffmpeg')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def probe(p):   # (width, height, fps, frames) of a video
    import cv2
    c = cv2.VideoCapture(str(p)); r = (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)), c.get(cv2.CAP_PROP_FPS), int(c.get(cv2.CAP_PROP_FRAME_COUNT))); c.release()
    return r


V, O = Path(a.videos), Path(a.out); (O / 'videos').mkdir(parents=True, exist_ok=True)
rows, missing = [], []
for l in open(B / 'generation_inputs.jsonl'):
    g = json.loads(l); name = f"{g['clip_id']}.{g['prompt_group']}.mp4"; src = V / name
    if not src.exists(): missing.append(name); continue
    w, h, fps, _ = probe(src); dst = O / 'videos' / name
    if (w, h) == (g['width'], g['height']): shutil.copyfile(src, dst)
    elif not dst.exists():
        subprocess.run([ff, '-y', '-v', 'error', '-i', str(src), '-vf', f"scale={g['width']}:{g['height']}:flags=lanczos", '-r', f'{fps:g}', '-an',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(dst)], check=True)
    W_, H_, fps2, n = probe(dst)
    if (W_, H_) != (g['width'], g['height']): sys.exit(f'{dst}: {W_}x{H_}, expected {g["width"]}x{g["height"]}')
    cond = B / g['condition_image']
    rows.append(dict(id=f"{a.model}__{g['clip_id']}__{g['prompt_group']}__s0", model=a.model, clip_id=g['clip_id'], prompt_group=g['prompt_group'], seed=0,
                     video=str(dst.resolve()), video_sha256=sha(dst), condition_image=str(cond), condition_sha256=sha(cond), condition_frame_index=0,
                     prompt=g['prompt'], fps=fps2, frames=n, width=W_, height=H_, source_video=str(src.resolve())))
with open(O / 'manifest.jsonl', 'w') as o: o.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
print(f'{len(rows)} videos in {O / "manifest.jsonl"}' + (f'; {len(missing)} missing, e.g. {missing[:5]}' if missing else ''))
