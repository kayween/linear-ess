import math

import numpy as np
import torch

import warnings

import time


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

        # Some common tensors that will be used more than once
        batch = x.size(-2)
        self.zeros = x.new_zeros((batch, 1))
        self.ones = x.new_ones((batch, 1))
        self.indices_batch = torch.arange(batch, dtype=torch.int64, device=x.device)

        alpha, beta = self.intersection_angles(x @ A.T, z @ A.T, b.unsqueeze(-2))
        # alpha, beta = self.intersection_angles(
        #     (x @ A.T).to(torch.float64),
        #     (z @ A.T).to(torch.float64),
        #     b.unsqueeze(-2).to(torch.float64),
        # )

        left, right = self.active_angles(alpha, beta)
        # self.left = self.left.to(torch.float64)
        # self.right = self.right.to(torch.float64)

        self.left, self.right = self.postprocess_active_angles(left, right)

        # self.csum = self.right.sub(self.left).clamp(min=0.).to(torch.float64).cumsum(dim=-1)
        self.csum = self.right.sub(self.left).clamp(min=0.).cumsum(dim=-1)

    def postprocess_active_angles(self, left, right):
        gap = torch.clamp(right - left, min=0.)

        epsilon = torch.tensor(1e-6, dtype=left.dtype, device=left.device)
        epsilon = torch.minimum(gap * 0.25, epsilon)

        left = left + epsilon
        right = right - epsilon

        # return left.to(torch.float64), right.to(torch.float64)
        return left, right

    def sample_angle(self):
        # u = self.csum[:, -1] * torch.rand(self.right.size(-2), dtype=torch.float64, device=self.x.device)
        u = self.csum[:, -1] * torch.rand(self.right.size(-2), device=self.x.device)

        # The returned index i satisfies self.csum[i - 1] < u <= self.csum[i]
        idx = torch.searchsorted(self.csum, u.unsqueeze(-1)).squeeze(-1)

        # Do a zero padding so that padded_csum[i] = csum[i - 1]
        # padded_csum = torch.cat([self.zeros.to(torch.float64), self.csum], dim=-1)
        padded_csum = torch.cat([self.zeros, self.csum], dim=-1)

        theta = u - padded_csum[self.indices_batch, idx] + self.left[self.indices_batch, idx]

        # def check_feasibility(theta):
        #     point = self.x * torch.cos(theta).unsqueeze(-1) + self.z * torch.sin(theta).unsqueeze(-1)
        #     if torch.max(point @ A.T - b).gt(0.).any():
        #         print("fuck")
        #         import ipdb; ipdb.set_trace()

        # check_feasibility(theta.to(self.b.dtype))

        return theta.to(self.b.dtype)
        # return ret
        # return ret

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

        # In extreme cases, arctan has NaN because the denominator (radius + p) is zero.
        # If this happens, q has to be zero and thus we rewrite arctan as zero.
        arctan[arctan.isnan().nonzero(as_tuple=True)] = 0.

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

    def active_angles(self, alpha, beta):
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
        srted, indices = torch.sort(alpha, descending=False)
        cummax = beta[self.indices_batch.unsqueeze(-1), indices].cummax(dim=-1).values

        srted = torch.cat([srted, self.ones * 2 * math.pi], dim=-1)
        cummax = torch.cat([self.zeros, cummax], dim=-1)
 
        return cummax, srted


class TruncatedGaussianSampler(object):
    def __init__(self, A, b):
        self.A = A
        self.b = b

        self.cnt_warnings = 0

    def next(self, x):
        nu = torch.randn(x.size(), dtype=x.dtype, device=x.device)

        sampler = EllipticalSliceSampler(self.A, self.b, x, nu)

        new_x =  sampler.sample_slice()

        if torch.max(new_x @ A.T - b).max().ge(0.):
            warnings.warn("sample violates the constraint")
            self.cnt_warnings += 1

            new_x = x

        return new_x


if __name__ == "__main__":
    device = "cuda:0"

    torch.set_default_dtype(torch.float64)
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

    # A = torch.randn(m, d, device=device)
    # x = torch.randn(1000, d, device=device)

    # prod = x @ A.T
    # b = prod.max(dim=-2).values + torch.rand(m, device=device)

    sampler = TruncatedGaussianSampler(A, b)

    start = time.time()
    for i in range(10000):
        # print("=== iter {:d} ===".format(i))
        # if i == 1400:
        #     # print(time.time() - start)
        #     break
        x = sampler.next(x)

    print(sampler.cnt_warnings)
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
