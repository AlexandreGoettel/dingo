# Asimov end-to-end tests

This directory contains the blueprints for the full end-to-end (e2e) CI test of
the asimov integration (`dingo/asimov/asimov.py`), run by
`.github/workflows/asimov-e2e.yml`.

The test runs `asimov apply` -> `asimov manage build submit` -> `asimov
monitor` against a real HTCondor mini-cluster inside the `htcondor/mini`
container, for asimov 0.7 (PyPI) and 0.8 (git `main`), mirroring the end-to-end
CI used by the asimov and bilby_pipe repositories.

It is deliberately fully self-contained:

- no BayesWave PSD stage and no real strain data / GWDataFind: dingo_pipe's
  Gaussian-noise simulation mode (`gaussian noise: true` in the event
  blueprint) is used instead;
- the DINGO network (a small toy posterior model, trained with the settings in
  `examples/toy_npe_model`) is expected as a workflow input named `model`
  (upload the `.pt` file manually in the GitHub UI when dispatching the
  workflow, or push it to the branch).

## The toy model

The e2e blueprints assume a network trained with
`examples/toy_npe_model/train_settings.yaml`, i.e. with a
`UniformFrequencyDomain` domain with `f_min = 20`, `f_max = 1024`,
`delta_f = 0.25` (4 s segment) and detectors `H1, L1`. Upload the trained
`model_latest.pt` (and, for GNPE models, `model_init_latest.pt`) as the
workflow input `model`. For a plain NPE model, set `model init` in the analysis
blueprint to `None`.
