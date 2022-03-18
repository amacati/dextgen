# Master Thesis Martin Schuck

[![PEP8 Check](https://github.com/amacati/rl/actions/workflows/linting.yaml/badge.svg)](https://github.com/amacati/rl/actions/workflows/linting.yaml)
[![Tests](https://github.com/amacati/rl/actions/workflows/testing.yaml/badge.svg)](https://github.com/amacati/rl/actions/workflows/testing.yaml)

## Clear saves folder from backups

Run the following command from the RL root folder.

> :warning: **WARNING**: Make sure you execute from the correct folder or this may have catastrophic consequences!

```$ find . -type d -name backup -exec rm -r "{}" \;```
