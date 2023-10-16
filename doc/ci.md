# Continuous Integration (CI)

This repository has access to a pool of FPGAs in Azure Pipelines for running
continuous integration tests on.

See [azure-pipelines.yml](../ci/azure-pipelines.yml) for the Azure pipeline
that currently runs on each pull request.

## Selecting FPGAs

The `FPGA SCA` Azure agent pool contains both a CW305 and CW310 agent.

To run a pipeline job on a particular FPGA, you must specify the board type in
the YAML specification file:

```yaml
jobs:
- job: some_cw305_job
  pool:
    name: FPGA SCA
    demands: BOARD -equals cw305
  steps:
    - ...

- job: some_cw310_job
  pool:
    name: FPGA SCA
    demands: BOARD -equals cw310
  steps:
    - ...
```

## Approving external CI runs

The CI pipelines run on systems managed by lowRISC. To prevent external GitHub
users from running arbitrary code on our systems, CI will not run on pull
requests from forks of users outside the [lowRISC GitHub organisation].

Members of the lowRISC GitHub organisation can manually allow a pull request to
run in CI by [adding a comment][Azure comment triggers]:

> /AzurePipelines run

[Azure comment triggers]: https://learn.microsoft.com/en-us/azure/devops/pipelines/repos/github?view=azure-devops&tabs=yaml#comment-triggers
[lowRISC GitHub organisation]: https://github.com/lowrisc/
