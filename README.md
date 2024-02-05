# Physics-Informed Neural Networks for Quantum Eigenvalue Problems

This repository contains code for the reproduction of some
of the experiments in [this](https://arxiv.org/abs/2203.00451) paper
with the same name.

The major improvement is the use of L-BFGS instead of Adam, which
showed to work better in the "Physics-Informed Neural Network" setting.

### Repo

The repo code for the reproductions of the paper mentioned
above is available under `submission/extension_task`. Whereas
the in the directory `submission/base_task` contains code
for slightly less difficult problems.

The file `submission/report.pdf` contains a description
of the report, mainly created as a submission for the course
Deep Learning in Scientific computing at ETH Zurich.
