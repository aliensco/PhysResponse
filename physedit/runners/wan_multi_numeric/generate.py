"""Wan2.2 TI2V 5B Multi numerical I2V generation with the training-time encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open

from .wan_numeric_conditioning import NumericBatch
from .wan_numeric_production_bridge import WanNumericConditioner
from .wan_numeric_runtime import get_current, set_current_from_prompt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NumericTextProjection(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, conditioner: WanNumericConditioner):
        super().__init__()
        self.base = base
        self.conditioner = conditioner
        self.active_positive = False
        self.positive_calls = 0
        self.negative_calls = 0
        self.last_residual_norm = 0.0

    def forward(self, context):
        projected = self.base(context)
        if not self.active_positive:
            self.negative_calls += 1
            return projected
        current = get_current()
        count = len(current["token_indices"])
        if projected.shape[0] != 1 or count == 0:
            raise RuntimeError("numeric I2V requires one prompt with supported values")
        # DiffSynth's VRAM manager moves the base projection independently of
        # the newly attached module. Follow the active DiT device on each call.
        if next(self.conditioner.parameters()).device != projected.device:
            self.conditioner.to(projected.device)
        batch = NumericBatch(
            values=torch.tensor(current["values"], device=projected.device, dtype=torch.float32).view(1, count, 1),
            type_ids=torch.tensor(current["type_ids"], device=projected.device, dtype=torch.long).view(1, count),
            token_indices=torch.tensor(current["token_indices"], device=projected.device, dtype=torch.long).view(1, count),
            component_mask=torch.ones((1, count, 1), device=projected.device, dtype=torch.float32),
        )
        conditioned = self.conditioner(projected, batch).context
        self.last_residual_norm = float((conditioned.float() - projected.float()).norm().item())
        if self.last_residual_norm <= 0:
            raise RuntimeError("Harmonic encoder produced no numeric residual")
        self.positive_calls += 1
        return conditioned


def load_pipeline(base_dir: Path, adapter: Path, expected_sha256: str | None):
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    if expected_sha256 and sha256(adapter) != expected_sha256:
        raise ValueError("adapter SHA256 mismatch")
    shards = sorted(base_dir.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if len(shards) != 3:
        raise FileNotFoundError("expected the three local Wan2.2 TI2V 5B DiT shards")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=str(base_dir / "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=[str(path) for path in shards]),
            ModelConfig(path=str(base_dir / "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=str(base_dir / "google" / "umt5-xxl")),
        redirect_common_files=False,
    )
    with safe_open(str(adapter), framework="pt", device="cpu") as handle:
        names = list(handle.keys())
        prefix = "text_embedding.conditioner."
        numeric = {name[len(prefix):]: handle.get_tensor(name) for name in names if name.startswith(prefix)}
        lora = {name: handle.get_tensor(name) for name in names if not name.startswith(prefix)}
    if len(lora) != 600 or len(numeric) != 5:
        raise ValueError(f"unexpected weight families: LoRA={len(lora)}, Harmonic={len(numeric)}")
    pipe.load_lora(pipe.dit, state_dict=lora, alpha=1, hotload=False)
    conditioner = WanNumericConditioner(model_dim=3072, max_components=1, num_types=8,
                                        harmonics=8, hidden_dim=512)
    conditioner.load_state_dict(numeric, strict=True)
    projection = NumericTextProjection(pipe.dit.text_embedding, conditioner)
    pipe.dit.text_embedding = projection
    original_model_fn = pipe.model_fn

    def conditioned_model_fn(*args, **kwargs):
        projection.active_positive = kwargs.get("positive") is True
        try:
            return original_model_fn(*args, **kwargs)
        finally:
            projection.active_positive = False

    pipe.model_fn = conditioned_model_fn
    return pipe, projection


def feature_delta(projection, tokenizer, prompt_a: str, prompt_b: str) -> dict:
    """Check learned Harmonic features for fixed text structure and changed values."""
    features = []
    bindings = []
    device = next(projection.conditioner.parameters()).device
    with torch.no_grad():
        for prompt in (prompt_a, prompt_b):
            bound = set_current_from_prompt(prompt, tokenizer)
            values = bound["numeric_values"]
            count = len(values)
            batch = NumericBatch(
                values=torch.tensor(values, device=device, dtype=torch.float32).view(1, count, 1),
                type_ids=torch.tensor(bound["numeric_type_ids"], device=device, dtype=torch.long).view(1, count),
                token_indices=torch.tensor(bound["numeric_token_indices"], device=device, dtype=torch.long).view(1, count),
                component_mask=torch.ones((1, count, 1), device=device),
            )
            features.append(projection.conditioner.encoder(batch).float().cpu())
            bindings.append(bound)
    if features[0].shape != features[1].shape:
        raise ValueError("comparison prompts must have identical token structure")
    difference = float((features[0] - features[1]).norm().item())
    if difference <= 0:
        raise RuntimeError("Harmonic features did not change with physical values")
    return {"feature_delta_l2": difference, "bindings": bindings}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256")
    parser.add_argument("--image", type=Path, required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--compare-prompt", help="Same structure, changed values; validates Harmonic features")
    parser.add_argument("--compare-prompt-file", type=Path)
    args = parser.parse_args(argv)
    prompt = args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else args.prompt
    compare_prompt = (args.compare_prompt_file.read_text(encoding="utf-8").strip()
                      if args.compare_prompt_file else args.compare_prompt)
    if args.frames < 5 or (args.frames - 1) % 4:
        parser.error("frames must be at least 5 and equal to 4n+1")
    pipe, projection = load_pipeline(args.base_dir, args.adapter, args.adapter_sha256)
    tokenizer = pipe.tokenizer.tokenizer
    comparison = feature_delta(projection, tokenizer, prompt, compare_prompt) if compare_prompt else None
    bindings = set_current_from_prompt(prompt, tokenizer)
    image = Image.open(args.image).convert("RGB").resize((args.width, args.height))
    video = pipe(prompt=prompt, negative_prompt=args.negative_prompt,
                 input_image=image, seed=args.seed, width=args.width, height=args.height,
                 num_frames=args.frames, num_inference_steps=args.steps,
                 cfg_scale=args.guidance, sigma_shift=args.shift, tiled=True)
    if projection.positive_calls < 1:
        raise RuntimeError("generation did not invoke the Harmonic encoder")
    from diffsynth.utils.data import save_video
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(args.output), fps=args.fps, quality=5)
    evidence = {
        "output": str(args.output), "output_sha256": sha256(args.output),
        "adapter_sha256": sha256(args.adapter), "seed": args.seed,
        "width": args.width, "height": args.height, "frames": args.frames,
        "fps": args.fps, "steps": args.steps, "guidance": args.guidance,
        "shift": args.shift, "bindings": bindings,
        "harmonic_positive_calls": projection.positive_calls,
        "harmonic_negative_calls": projection.negative_calls,
        "last_numeric_residual_l2": projection.last_residual_norm,
        "comparison": comparison,
    }
    receipt = args.output.with_suffix(".receipt.json")
    receipt.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
