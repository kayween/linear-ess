import math

import torch

from src.ess import EllipticalSliceSampler
from botorch.utils.probability.lin_ess import \
    LinearEllipticalSliceSampler as BotorchEllipticalSliceSampler

import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    torch.manual_seed(0)

    # device = "cpu"

    # d = 100
    # lst_num_constraints = [100, 500, 1000, 1500, 2000, 2500, 3000]

    device = "cuda:0"
    # torch.set_default_dtype(torch.float64)

    mat = torch.eye(20000, device="cuda:0")
    mat = torch.sum(mat @ mat)

    d = 1000
    lst_num_constraints = [1000, 1000, 2000, 3000, 4000, 5000]

    lst_time_ess = []
    lst_time_botorch = []

    for m in lst_num_constraints:
        A = torch.randn(m, d, device=device) / math.sqrt(m)
        x = torch.randn(1, d, device=device) / math.sqrt(d)

        prod = x @ A.T
        b = prod.max(dim=-2).values + torch.rand(m, device=device)

        num_steps = n = 1000

        # Let's start sampling
        start = time.time()

        sampler = EllipticalSliceSampler(A, b, x, burnin=0)
        samples = sampler.launch(num_steps=num_steps, thinning=None)

        end = time.time()

        print("running time {:.4f}".format(end - start))
        print("count violations {:d}".format(sampler.cnt_violations))

        lst_time_ess.append(end - start)

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

    plt.figure(figsize=(3, 2.5))
    plt.plot(lst_num_constraints[1:], lst_time_ess[1:], marker='^', label='our ess')
    plt.plot(lst_num_constraints[1:], lst_time_botorch[1:], marker='s', label='botorch ess')
    plt.title('CPU Running Time' if device == "cpu" else "GPU Running Time")
    plt.ylabel('time (s)')
    plt.xlabel('num. of constraints')
    plt.legend()
    plt.tight_layout()
    plt.show()
