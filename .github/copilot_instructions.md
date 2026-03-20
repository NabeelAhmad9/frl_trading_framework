# GitHub Copilot Instructions for `frl-trading-framework`

## Project Objective
- Implement a Forex trading framework for RL research with realistic market modeling, modular rewards, and reproducible experiments.
- Support DQN and Double DQN agents.
- Enforce anti-lookahead behavior, strict legal-action masking, and risk-aware trading.

## Environment Management & Dependencies

- Always use the dedicated Conda environment `torch` for running scripts, tests, training, evaluation, or installing packages.
- Never run code from the `base` environment.

### Activating the Environment

- Activate `torch` at the start of any workflow:
  ```bash
  conda activate torch
  ```
- Verify the correct environment is active:
  ```bash
  conda info --envs
  ```

### Installing Libraries

- Always install packages into the `torch` environment:
  ```
  conda install <package> -n torch
  # or
  pip install <package>  # only after activating torch
  ```
- Do not modify the `base` environment; it should remain untouched.

### Running Scripts, Tests, and Programs

- Execute all scripts, tests, and programs only from the `torch` environment.

- python -n torch your_script.py