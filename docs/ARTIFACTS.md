# Data and model artifacts

| Artifact | Location | Status |
|---|---|---|
| Simulation videos, prompts, annotations, and 5,200 Core100 outputs | [PhysResponse dataset](https://huggingface.co/datasets/SunYat-senUniversity/PhysResponse) | Public dataset |
| Frozen Core100 pair judgments and expected scores | [Evaluation files](https://huggingface.co/datasets/SunYat-senUniversity/PhysResponse/tree/main/core100/evaluation) | Public frozen judgments and expected scores |
| Core100 inversion evaluation base | [Standard release](https://github.com/aliensco/PhysResponse/releases/tag/evaluation-base-standard-v1) | 50 parts and SHA-256 manifest. See the [download instructions](EVALUATE.md#parameter-inversion). |
| SAM 3.1 tracker | [Source](https://github.com/facebookresearch/sam3) and [checkpoint](https://huggingface.co/facebook/sam3.1) | Obtain from the upstream providers under their access terms. |

This source snapshot includes a CPU smoke-test fixture and no checkpoint. GPU tracking uses the upstream checkpoint.

Exact compact-prompt token counts require the [upstream SentencePiece vocabularies](../physedit/prompt_tokenizers/README.md).
