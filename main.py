import math

import numpy as np
import torch


class AngleSampler(object):
    def __init__(self, A, b, x, nu):
        """
        Linear inequality constraints
            A @ x + b <= 0.

        Args:
            A: tensor
            b: tensor
            x: tensor, has to satisfies the linear inequality constraints
            nu: tensor
        """
        if not A.matmul(x).add(b).le(0.).all():
            raise RuntimeError

        angles = self.feasible_angles(A @ x, A @ nu, b)

        if not angles.size % 2 == 0:
            raise RuntimeError

        # each row of self.slices represents an interval [a, b]
        self.slices = angles.reshape(-1, 2)
        print(self.slices)

        self.csum = np.cumsum(self.slices[:, 1] - self.slices[:, 0])

    def sample(self):
        u = self.csum[-1] * np.random.rand()
        idx = np.searchsorted(self.csum, u)
        return self.slices[idx, 0] + u - (self.csum[idx - 1] if idx > 0 else 0.)

    def feasible_angles(self, p, q, bias):
        """
        Solve the trigonometry inequalities
            p * cos(theta) + q * sin(theta) + bias <= 0.
        """
        normalize = torch.sqrt(p ** 2 + q ** 2)

        # It's impossible that the ratio < -1 if A @ x + b <= 0.
        # Thus, () <= 1 is equivalent to abs() <= 1.
        has_solution = -bias / normalize <= 1.

        if not has_solution.any():
            return np.array([0., 2 * math.pi])

        p, q, bias, normalize = (
            p[has_solution], q[has_solution], bias[has_solution], normalize[has_solution]
        )

        arccos = torch.arccos(-1. * bias / normalize)
        arctan = torch.arctan(q / (normalize + p))

        theta1 = -1. * arccos + 2. * arctan
        theta2 = +1. * arccos + 2. * arctan

        # translate every angle to [0, 2 * pi]
        theta1 = theta1 + theta1.lt(0.) * 2. * math.pi
        theta2 = theta2 + theta2.lt(0.) * 2. * math.pi

        theta1, theta2 = torch.minimum(theta1, theta2), torch.maximum(theta1, theta2)

        sorted_one, indices_one = torch.sort(theta1, descending=False)
        cummax = theta2[indices_one].cummax(dim=-1).values
        keep_theta_one = sorted_one > torch.cat([torch.tensor([-1.]), cummax[:-1]], dim=-1)

        sorted_two, indices_two = torch.sort(theta2, descending=True)
        cummin = theta1[indices_two].cummin(dim=-1).values
        keep_theta_two = sorted_two < torch.cat([torch.tensor([2 * math.pi + 1.]), cummin[:-1]], dim=-1)

        # print(sorted_one[keep_theta_one])
        # print(sorted_two[keep_theta_two])

        unsorted_list = [0.] + [2. * math.pi] \
            + sorted_one[keep_theta_one].tolist() \
            + sorted_two[keep_theta_two].tolist()

        return np.array(sorted(unsorted_list))


class TruncatedGaussianSampler(object):
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def next(self, x):
        nu = torch.randn(x.size(-1))

        sampler = AngleSampler(self.A, self.b, x, nu)

        theta = sampler.sample()
        return x * math.cos(theta) + nu * math.sin(theta)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # domain is -1 <= x <= 3
    A = torch.tensor([[-1.], [1.]])
    b = torch.tensor([-1., -3.])

    x = torch.tensor([0.])

    sampler = TruncatedGaussianSampler(A, b)

    samples = []

    for i in range(400):
        for j in range(100):
            x = sampler.next(x)

        samples.append(x)

    samples = torch.cat(samples, dim=-1)
    mean = samples.mean()
    std = samples.var()

    print(mean, std)

    from scipy.stats import truncnorm
    mean, var = truncnorm.stats(-1, 3)
    print(mean, var)

    import matplotlib.pyplot as plt
    plt.hist(samples.tolist(), density=True, bins=20)
    # plt.hist(samples.tolist(), cumulative=True, density=True, bins=20)

    xx = np.linspace(-1, 3.)
    # plt.plot(xx, truncnorm.cdf(xx, -1, 3), label='truncated Gaussian CDF')
    plt.plot(xx, truncnorm.pdf(xx, -1, 3), label='truncated Gaussian PDF')
    plt.legend()

    # plt.title('cumulative histogram of MCMC samples')
    plt.title('density histogram of MCMC samples')
    plt.show()
