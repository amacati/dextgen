# Dext-Gen: Dexterous Grasping in Sparse Reward Environments with Full Orientation Control 

[![PEP8 Check](https://github.com/amacati/rl/actions/workflows/linting.yaml/badge.svg)](https://github.com/amacati/rl/actions/workflows/linting.yaml)
[![Tests](https://github.com/amacati/rl/actions/workflows/testing.yaml/badge.svg)](https://github.com/amacati/rl/actions/workflows/testing.yaml)

In order to run the code from this project, install anaconda and create a conda environment from the provided [environment file](environment.yaml) with

```
user@pc: $ conda create -f environment.yaml
```
, activate the new environment with

```
user@pc: $ conda activate dextgen
```

and install the dextgen packages from the project root folder with

```
user@pc: $ pip install .
```

Instructions on how to use the project are available in the [package documentation](docs/). To read the documentation, build it from the [docs directory](docs/) with

```
user@pc: $ make html
```

and open the index file under 'docs/_build/html/index.html' in the browser of your choice, e.g. with

```
user@pc: $ firefox _build/html/index.html
```

The experiment environments are located in the [envs](envs/) module, the learning algorithm is located under [mp_rl](mp_rl/). The grasp optimization routines are organized in the [optim](optim/) module.
Credit to code from external repositories is given in the individual source files.

If you have any issues with the installation of MuJoCo and mujoco-py such as the osmesa.h error, see the [instructions](mujoco_install.md) on how to resolve common issues.