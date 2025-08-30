# Continuous Integration (CI)

This repository has access to a pool of machines that have FPGAs attached and
can be used to run GitHub Actions continuous integration jobs on.

See [.github/workflows/fpga.yml](../.github/workflows/fpga.yml) for a GitHub
Actions workflow that uses these FPGAs and runs on each pull request.

## Selecting FPGAs

The pool of FPGAs contain CW310 runners.

To run a job on a machine with a particular FPGA, you must specify the `cw310` labels in addition to `ubuntu-22.04-fpga` in the YAML
specification file:

```yaml
jobs:
  some_cw310_job:
    runs-on: [ubuntu-22.04-fpga, cw310]
    steps:
      - ...
```

## Approving external CI runs

The CI pipelines run on systems managed by lowRISC. To prevent external GitHub
users from running arbitrary code on our systems, CI will not run on pull
requests from forks of users outside the [lowRISC GitHub organisation].

GitHub users with write access to the repository can allow a pull request to run
in CI using the GitHub UI. A button should appear next to the list of checks
that will be run.

[lowRISC GitHub organisation]: https://github.com/lowrisc/
