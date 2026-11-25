<div align="center">
  <img src="docs/assets/physresponse-logo.png" width="420" alt="PhysResponse">
  <h1>Probing the Physics of Video Generation<br>with Controlled Interventions</h1>
  <p>
    <a href="https://arxiv.org/abs/2611.24419"><img src="docs/assets/arxiv.svg" height="26" alt="arXiv"></a>
    <a href="https://huggingface.co/datasets/SunYat-senUniversity/PhysResponse"><img src="docs/assets/data.svg" height="26" alt="Hugging Face dataset"></a>
    &nbsp;·&nbsp; <a href="#install">Quickstart</a>
    &nbsp;·&nbsp; <a href="#evaluate">Evaluation</a>
  </p>
  <p>
    <img src="docs/assets/sysu.png" height="42" alt="Sun Yat-sen University">
    &nbsp;&nbsp;
    <img src="docs/assets/mbzuai.png" height="42" alt="Mohamed bin Zayed University of Artificial Intelligence">
    &nbsp;&nbsp;
    <img src="docs/assets/hkust.png" height="42" alt="The Hong Kong University of Science and Technology">
  </p>
</div>

PhysResponse tests how video generators respond to controlled changes in gravity, friction, and initial speed. The benchmark recovers effective physical parameters from generated motion and checks event outcomes on either side of a critical transition.

| Simulation videos | Interaction types | Prompt bindings | Core100 cases |
| :---: | :---: | :---: | :---: |
| **72,000** | **20** | **288,000** | **100** |

<p align="center">
  <img src="docs/assets/t09_boundary.jpg" width="740" alt="Core100 reference frames showing distinct book-stack outcomes under two gravity levels">
  <br><em>Core100 reference cases T09_C02 and T09_C03. The upper book stays on the stack at 3.71 m/s² and lands on the table at 9.81 m/s².</em>
</p>

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
