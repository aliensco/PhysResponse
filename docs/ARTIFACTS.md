# Data and model artifacts

| Artifact | Location | Status |
|---|---|---|
| PhysResponse simulation data | [Hugging Face](https://huggingface.co/datasets/phy2/PhysResponse) | Private release candidate |
| Prompt dataset | [Hugging Face](https://huggingface.co/datasets/phy2/PhysResponse-Prompts) | Private release candidate |
| Core100 generated videos and frozen pair-scoring inputs | [Hugging Face](https://huggingface.co/datasets/SunYat-senUniversity/PhysResponse-Core100-Generations/tree/release-v1-rc/release_candidate) | Private release candidate |
| Full inversion evaluation base and tracking assets | Separate distribution | Public download awaits asset and license clearance |

This source snapshot includes a CPU smoke-test fixture and no checkpoint. The last artifact is required to reproduce the full inversion experiment.

Exact compact-prompt token counts require the [upstream SentencePiece vocabularies](../physedit/prompt_tokenizers/README.md).
