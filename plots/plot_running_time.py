import sys
import math

import torch

from src.ess import EllipticalSliceSampler
from botorch.utils.probability.lin_ess import \
    LinearEllipticalSliceSampler as BotorchEllipticalSliceSampler

import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cpu" if sys.argv[1] == "cpu" else "cuda:0"

    print(device)

    if device == "cpu":
        lst_dims = [200, 400, 600, 800, 1000, 1200, 1600, 1800, 2000]
    else:
        lst_dims = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

    # warm up
    mat = torch.eye(20000, device="cuda:0")
    mat = torch.sum(mat @ mat)

    lst_time_botorch = []
    lst_time_ess = []
    lst_time_parallel_ess = []

    for d in lst_dims:
        print("****** dimension = {:d} ******".format(d))

        m = d
        A = torch.randn(m, d, device=device) / math.sqrt(m)
        x = torch.randn(1, d, device=device) / math.sqrt(d)

        prod = x @ A.T
        b = prod.squeeze() + torch.rand(m, device=device)

        # number of samples
        n = 1000

        # BoTorch's ess
        start = time.time()
        botorch_sampler = BotorchEllipticalSliceSampler(
            inequality_constraints=(A, b.unsqueeze(-1)),
            interior_point=x.transpose(-1, -2),
            check_feasibility=False,
            burnin=0,
            thinning=0,
        )
        botorch_sampler.draw(n=n)
        end = time.time()

        print("botorch running time {:.4f}".format(end - start))
        lst_time_botorch.append(end - start)

        # Our ess
        start = time.time()
        sampler = EllipticalSliceSampler(A, b, x, burnin=0)
        samples = sampler.launch(num_steps=n, thinning=None)
        end = time.time()

        print(
            "a single chain running time {:.4f},".format(end - start),
            "cnt violations {:d}".format(sampler.cnt_violations),
        )
        lst_time_ess.append(end - start)

        # Our ess with 10 chains running in parallel
        start = time.time()
        sampler = EllipticalSliceSampler(
            A, b, x.expand(10, -1).clone(), burnin=0,
        )
        samples = sampler.launch(num_steps=n // 10, thinning=None)
        end = time.time()

        print(
            "10 chains running time {:.4f},".format(end - start),
            "cnt violations {:d}".format(sampler.cnt_violations),
        )
        lst_time_parallel_ess.append(end - start)

    plt.figure(figsize=(3, 2.5))
    plt.plot(lst_dims, lst_time_botorch, marker='s', label='botorch ess')
    plt.plot(lst_dims, lst_time_ess, marker='^', label='ours (1 chain)')
    plt.plot(lst_dims, lst_time_parallel_ess, marker='^', label='ours (10 chains)')
    plt.title('CPU Running Time' if device == "cpu" else "GPU Running Time")
    plt.ylabel('time (s)')
    plt.xlabel('num. of dimensions')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout(pad=0)
    plt.show()
