# Elliptical Slice Sampling for Linearly Truncated Multivariate Normal Distributions

This repository implements an improved elliptical slice sampling for truncated multivariate normal distributions described in the paper "[A Fast, Robust Elliptical Slice Sampling Implementation for Linearly Truncated Multivariate Normal Distributions](https://arxiv.org/abs/2407.10449)".

**Updates**
- [12/2024] This paper is presented at the BDU workshop at NeurIPS 2024.
- [07/2024] This implementation is also available in BoTorch since v0.11.3 at [here](https://github.com/pytorch/botorch/blob/da28b4315fb5e6d09fbc572e56f7a89da004d842/botorch/utils/probability/lin_ess.py#L47).

**TODOs**
- [ ] Add Gibbs sampling.
- [ ] Add separation of variables for estimating multivariate normal probabilities.

## Run the Code
To run the code, you will need to install the package by running the following command. 
```
pip install -e .
```

Refer to `./notebooks/` for an example on using elliptical slice sampling for Bayesian probit regression.
