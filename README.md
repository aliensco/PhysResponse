# PhysResponse

Research code for **PhysResponse: Probing the Physics of Video Generation with Controlled Interventions**. The benchmark evaluates physical parameters inferred from motion and event outcomes under controlled interventions.

## Install

Python 3.12 is the supported release environment. Install the package in a virtual environment.

```sh
python -m pip install .
physedit --help
python -m physprobe.inversion.evaluate --help
```

For unit tests:

```sh
python -m pip install ".[test]"
python -m pytest -q
```

The included tests and synthetic scoring example run on CPU with no external dataset or GPU.

## Evaluate

- [Critical-event scoring and parameter inversion](docs/EVALUATE.md)
- [Dataset, generated-video and evaluation package status](docs/ARTIFACTS.md)

## Citation and license

Cite the [PhysResponse paper](https://arxiv.org/abs/2611.24419). The code in this repository is released under the [MIT License](LICENSE). Dataset files, annotations, and third-party model assets are distributed separately under their applicable terms.
