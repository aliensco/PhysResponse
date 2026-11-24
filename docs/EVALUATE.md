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

The full pipeline requires an unpacked Core100 evaluation base with reference videos, camera and scene bindings, and simulation libraries, plus an approved SAM source tree and checkpoint. See the [artifact statuses](ARTIFACTS.md) before running the example. GPU support is required for tracking.

```sh
export PHYSPROBE_WORK=/path/to/phyresponse_eval
python -m physprobe.inversion.make_manifest --videos VIDEO_DIR --model MODEL --out MANIFEST_DIR
python -m physprobe.inversion.evaluate --videos MANIFEST_DIR --name MODEL --sam3 SAM_SOURCE --checkpoint SAM_CHECKPOINT
```

Use `python -m physprobe.inversion.evaluate --help` for stage selection and optional tokenizer parameters.
