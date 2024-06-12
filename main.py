import math

import numpy as np
import torch

import warnings


class EllipticalSliceSampler(object):
    def __init__(self, A, b, x, z):
        """
        Linear inequality constraints
            A @ x + b <= 0.

        Args:
            A (tensor): the matrix of size (m, d) representing the constraint
            b (tensor): the vector of size (d,) representing the constraint
            x (tensor): a point on ellipse of size (*, d). Has to satisfy the linear inequality constraints
            z (tensor): another point on the ellipse of size (*, d)
        """
        self.A = A
        self.b = b

        self.x = x = x.unsqueeze(-2) if x.dim() == 1 else x
        self.z = z = z.unsqueeze(-2) if z.dim() == 1 else z

        if (x @ A.T - b).gt(0.).any():
            raise RuntimeError

        self.one_to_batch = torch.arange(x.size(-2), dtype=torch.int64, device=x.device)

        alpha, beta = self.intersection_angles(x @ A.T, z @ A.T, b.unsqueeze(-2))

        self.left, self.right = self.find_active_slices(alpha, beta)

        self.csum = self.right.sub(self.left).clamp(min=0.).cumsum(dim=-1)

    def sample_angle(self):
        u = self.csum[:, -1] * torch.rand(self.right.size(-2), device=self.x.device)

        idx = torch.searchsorted(self.csum, u.unsqueeze(-1)).squeeze(-1)
        is_gt_zero = idx > 0

        u[is_gt_zero] -= self.csum[is_gt_zero, idx[is_gt_zero] - 1]

        return self.left[self.one_to_batch, idx] + u

    def sample_slice(self):
        theta = self.sample_angle()
        point = self.x * torch.cos(theta).unsqueeze(-1) + self.z * torch.sin(theta).unsqueeze(-1)
        return point.squeeze(dim=-2)

    def intersection_angles(self, p, q, bias):
        """
        Solve the trigonometry inequalities
            p * cos(theta) + q * sin(theta) <= b.

        Args:
            p (tensor): of size (batch, m)
            q (tensor): of size (batch, m)
            b (tensor): broadcastable with p and q
        """
        radius = torch.sqrt(p ** 2 + q ** 2)

        if radius.abs().lt(1e-10).any():
            raise RuntimeError

        # It's impossible that the ratio < -1 if A @ x <= b.
        assert bias.div(radius).ge(-1).all()

        if bias.div(radius).min().le(-1 + 1e-6):
            warnings.warn(
                "The ellipse is almost outside the constraint. This may cause numerical issues."
            )

        has_solution = bias.div(radius).lt(1.)

        arccos = torch.arccos(bias / radius)
        arctan = torch.arctan(q / (radius + p))

        theta1 = -1. * arccos + 2. * arctan
        theta2 = +1. * arccos + 2. * arctan

        theta1[~has_solution] = 0.
        theta2[~has_solution] = 0.

        # translate every angle to [0, 2 * pi]
        theta1 = theta1 + theta1.lt(0.) * 2. * math.pi
        theta2 = theta2 + theta2.lt(0.) * 2. * math.pi

        alpha, beta = torch.minimum(theta1, theta2), torch.maximum(theta1, theta2)

        # return alpha, beta 
        alpha = alpha - torch.minimum(alpha * 0.01, torch.ones_like(alpha) * 1e-5)
        beta = beta + torch.minimum((2 * math.pi - beta) * 0.01, torch.ones_like(beta) * 1e-5)

        return alpha, beta

    def find_active_slices(self, alpha, beta):
        """
        Construct endpoints of active elliptical slices from intersection angles.

        Args:
            alpha: A tensor of size (batch, m) representing the smaller intersection angles
            beta: A tensor of size (batch, m) representing the larger intersection angles

        Return:
            A tuple (left, right) of tensors representing the active slices. Both tensors are of size (batch, m + 1).
            For the i-th batch and the j-th constraint, the interval from left[i, j] to right[i, j]
            is an active slice if and only if left[i, j] <= right[i, j].
        """
        batch = alpha.size(-2)

        ones = alpha.new_ones((batch, 1))
        zeros = beta.new_zeros((batch, 1))

        srted, indices = torch.sort(alpha, descending=False)
        cummax = beta[self.one_to_batch.unsqueeze(-1), indices].cummax(dim=-1).values

        srted = torch.cat([srted, ones * 2 * math.pi], dim=-1)
        cummax = torch.cat([zeros, cummax], dim=-1)
 
        return cummax, srted


class TruncatedGaussianSampler(object):
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def next(self, x):
        nu = torch.randn(x.size(), dtype=x.dtype, device=x.device)

        sampler = EllipticalSliceSampler(self.A, self.b, x, nu)

        return sampler.sample_slice()


if __name__ == "__main__":
    device = "cuda:0"

    # torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    np.random.seed(0)
    torch.manual_seed(0)

    # domain is -1 <= x <= 3
    A = torch.tensor([[-1.], [1.]], device=device)
    b = torch.tensor([1., 3.], device=device)
    x = torch.zeros(10000, 1, device=device)

    # m = 20
    # d = 10

    # A = torch.rand(m, d, device=device)
    # x = torch.randn(d, device=device)

    # b = A @ x + torch.rand(m, device=device)

    sampler = TruncatedGaussianSampler(A, b)

    for i in range(2000):
        print("=== iter {:d} ===".format(i))
        if i == 756:
            import ipdb; ipdb.set_trace()
        x = sampler.next(x)

    print("finish sampling")

    samples = x
    mean = samples.mean(dim=-2)
    std = samples.var(dim=-2)

    print(mean, std)

    from scipy.stats import truncnorm
    mean, var = truncnorm.stats(-1, 3)
    print(mean, var)

    # samples = samples.squeeze()

    # print(samples.max())
    # import matplotlib.pyplot as plt
    # # plt.hist(samples.tolist(), density=True, bins='auto')
    # plt.hist(samples.tolist(), cumulative=True, density=True, bins='auto')

    # xx = np.linspace(-1, 3.)
    # # plt.plot(xx, truncnorm.pdf(xx, -1, 3), label='truncated Gaussian PDF')
    # plt.plot(xx, truncnorm.cdf(xx, -1, 3), label='truncated Gaussian CDF')
    # plt.legend()

    # # plt.title('cumulative histogram of MCMC samples')
    # plt.title('density histogram of MCMC samples')
    # plt.show()
