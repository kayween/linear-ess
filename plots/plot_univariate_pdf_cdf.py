import numpy as np
import torch

from src.ess import EllipticalSliceSampler

import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    device = "cuda:0"

    torch.manual_seed(0)

    lb, ub = -1, 3
    # lb, ub = 6, 7

    # domain is lb <= x <= ub
    A = torch.tensor([[-1.], [1.]], device=device)
    b = torch.tensor([-lb, ub], device=device)
    x = torch.zeros(2000, 1, device=device) + 0.5 * (lb + ub)

    start = time.time()

    sampler = EllipticalSliceSampler(A, b, x, burnin=500)
    samples = sampler.launch(num_steps=500, thinning=10)

    end = time.time()

    print(samples.size())

    print("running time {:.4f}".format(end - start))
    print("count violations {:d}".format(sampler.cnt_violations))

    mean = samples.mean(dim=-2)
    std = samples.var(dim=-2)

    print(mean, std)

    from scipy.stats import truncnorm
    mean, var = truncnorm.stats(lb, ub)
    print(mean, var)

    xx = np.linspace(lb, ub)
    samples = samples.squeeze()

    plt.figure(figsize=(4, 3))
    plt.hist(samples.tolist(), density=True, bins='auto')
    plt.plot(xx, truncnorm.pdf(xx, lb, ub), label='ground truth PDF')
    plt.title('density histogram of MCMC samples')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(4, 3))
    plt.hist(samples.tolist(), cumulative=True, density=True, bins='auto')
    plt.plot(xx, truncnorm.cdf(xx, lb, ub), label='ground truth CDF')
    plt.title('cumulative histogram of MCMC samples')
    plt.legend()
    plt.tight_layout()
    plt.show()
