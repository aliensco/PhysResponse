# Data and model artifacts

| Artifact | Location | Status |
|---|---|---|
| Simulation videos, prompts, annotations, and 5,200 Core100 outputs | [PhysResponse dataset](https://huggingface.co/datasets/SunYat-senUniversity/PhysResponse) | Private release candidate |
| Frozen Core100 pair judgments and expected scores | [Evaluation files](https://huggingface.co/datasets/SunYat-senUniversity/PhysResponse/tree/main/core100/evaluation) | Included in the dataset release candidate |
| Full inversion evaluation base and tracking assets | Separate distribution | Public download awaits asset and license clearance |

This source snapshot includes a CPU smoke-test fixture and no checkpoint. The last artifact is required to reproduce the full inversion experiment.

Exact compact-prompt token counts require the [upstream SentencePiece vocabularies](../physedit/prompt_tokenizers/README.md).
