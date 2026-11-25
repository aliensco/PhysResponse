# Evaluation

## Core100 pair scores

The standalone scorer reads frozen binary endpoint judgments and reference labels. Each video pair has unit weight, split equally over its registered events. An event contributes to decided weight when both endpoints have labels. Its weight is correct when both labels match their references.

| Metric | Definition |
|---|---|
| PairAcc | Correct weight / decided weight |
| Coverage | Decided weight / 51 |
| VPSR | Correct weight / 51 |

Use the included synthetic example to verify the scoring path.

```sh
python score_pairs.py --input examples/pair_endpoints.synthetic.jsonl.gz --expected examples/pair_scores.synthetic.csv --output scores.csv
```

For the paper comparison, download the frozen judgments and expected scores from the [PhysResponse dataset](https://huggingface.co/datasets/SunYat-senUniversity/PhysResponse). Install the Hub CLI with `python -m pip install huggingface_hub`. If the dataset requires authentication, run `hf auth login` with an authorized account. For reproducibility, pass the dataset release commit SHA to `--revision`.

```sh
hf download SunYat-senUniversity/PhysResponse core100/evaluation/pair_endpoints.jsonl.gz core100/evaluation/expected_pair_scores.csv --repo-type dataset --local-dir evaluation-data
python score_pairs.py --input evaluation-data/core100/evaluation/pair_endpoints.jsonl.gz --expected evaluation-data/core100/evaluation/expected_pair_scores.csv --output scores.csv
```

The dataset indexes use `standard` for the Standard prompt format. The scorer reproduces pair metrics from those judgments. It does not infer event labels from pixels or compute NearAcc and FarAcc. For the registration, media validation, and scoring command options, run `physedit boundary --help`.

## Parameter inversion

Download the [Standard evaluation base](https://github.com/aliensco/PhysResponse/releases/tag/evaluation-base-standard-v1) from this repository. The release contains 50 numbered parts and a JSON checksum manifest (838,368,615 bytes combined). From the repository root, run the following commands with GitHub CLI, Python 3.12, and tar installed. If repository access requires authentication, run `gh auth login` first.

```sh
gh release download evaluation-base-standard-v1 --repo aliensco/PhysResponse --pattern "phyresponse_eval_standard.*" --dir inversion-assets
python scripts/assemble_inversion_base.py inversion-assets
tar -xzf inversion-assets/phyresponse_eval_standard.tar.gz -C inversion-assets
```

The assembly command checks every part and the complete archive. The extracted work directory is `inversion-assets/phyresponse_eval`. Use a short extraction path on Windows to accommodate long simulation-library filenames. Follow this guide for the current repository and dataset links.

Tracking requires a compatible GPU, the [SAM 3 source](https://github.com/facebookresearch/sam3), and the upstream [SAM 3.1 checkpoint](https://huggingface.co/facebook/sam3.1) obtained under its access terms. The checkpoint used for validation was `sam3.1_multiplex.pt` (SHA-256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`). Install the tracking dependencies in the GPU environment and make an FFmpeg build with libx264 available for input normalization.

Use the evaluation base with Standard prompt names. Name Standard input clips `<clip_id>.standard.mp4`, for example `T03_C01.standard.mp4`.

```sh
export PHYSPROBE_WORK="$(pwd)/inversion-assets/phyresponse_eval"
python -m physprobe.inversion.make_manifest --videos VIDEO_DIR --model MODEL --out MANIFEST_DIR
python -m physprobe.inversion.evaluate --videos MANIFEST_DIR --name MODEL --sam3 SAM_SOURCE --checkpoint SAM_CHECKPOINT
```

Use `python -m physprobe.inversion.evaluate --help` for stage selection and optional tokenizer parameters.
