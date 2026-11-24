"""Generate PhyResponse manifest rows with a Wan Multi LoRA and Harmonic encoder."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

from .generate import load_pipeline, sha256
from .wan_numeric_runtime import set_current_from_prompt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.generation_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        return 0
    pipe, projection = load_pipeline(args.base_dir, args.adapter, args.adapter_sha256)
    tokenizer = pipe.tokenizer.tokenizer
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    failed = 0
    for row in rows:
        raw = Path(row["generated_video"])
        resized = Path(row["generated_video_resized_to_source"])
        try:
            if raw.exists() or resized.exists():
                raise FileExistsError(f"refusing to overwrite existing media: {row['sample_id']}")
            if row.get("mode", "i2v") != "i2v":
                raise ValueError("Wan Multi runner requires I2V")
            prompt = Path(row["prompt_file"]).read_text(encoding="utf-8").strip()
            negative = Path(row["negative_prompt_file"]).read_text(encoding="utf-8").strip()
            binding = set_current_from_prompt(prompt, tokenizer)
            width = int(row.get("gen_width") or 832)
            height = int(row.get("gen_height") or 480)
            frames = int(row["num_frames"])
            fps = int(row["output_fps"])
            image = Image.open(row["condition_image"]).convert("RGB").resize((width, height))
            before_calls = projection.positive_calls
            video = pipe(prompt=prompt, negative_prompt=negative, input_image=image,
                         seed=int(row["seed"]), width=width, height=height, num_frames=frames,
                         num_inference_steps=args.steps, cfg_scale=args.guidance,
                         sigma_shift=args.shift, tiled=True)
            if projection.positive_calls <= before_calls:
                raise RuntimeError("Harmonic encoder was bypassed during generation")
            from diffsynth.utils.data import save_video
            raw.parent.mkdir(parents=True, exist_ok=True)
            partial = raw.with_name(raw.stem + ".partial.mp4")
            save_video(video, str(partial), fps=fps, quality=5)
            if not partial.is_file() or partial.stat().st_size == 0:
                raise RuntimeError("video encoder did not produce media")
            os.replace(partial, raw)
            resized.parent.mkdir(parents=True, exist_ok=True)
            resized_partial = resized.with_name(resized.stem + ".partial.mp4")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-vf",
                            f"scale={int(row['source_width'])}:{int(row['source_height'])}:flags=lanczos",
                            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(resized_partial)],
                           check=True)
            os.replace(resized_partial, resized)
            receipt = {"sample_id": row["sample_id"], "raw_sha256": sha256(raw),
                       "resized_sha256": sha256(resized), "adapter_sha256": args.adapter_sha256,
                       "binding": binding, "harmonic_positive_calls": projection.positive_calls,
                       "last_numeric_residual_l2": projection.last_residual_norm}
            (Path(row["out_dir"]) / "wan_multi_numeric_receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"sample_id": row["sample_id"], "status": "completed"}), flush=True)
        except Exception as exc:
            failed += 1
            print(json.dumps({"sample_id": row.get("sample_id"), "status": "failed",
                              "error": repr(exc), "traceback": traceback.format_exc()}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
