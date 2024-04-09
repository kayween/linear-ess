import math

import numpy as np
import torch


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

        if (x @ A.T + b).gt(0.).any():
            raise RuntimeError

        self.one_to_batch = torch.arange(x.size(-2), dtype=torch.int64, device=x.device)

        angles = self.intersection_angles(x @ A.T, z @ A.T, b.unsqueeze(-2))

        self.right_endpoints, self.left_endpoints = self.candidate_endpoints(angles)
        self.csum = self.right_endpoints.sub(self.left_endpoints).clamp(min=0.).cumsum(dim=-1)

        # print("right endpoints\n", self.right_endpoints)
        # print("left  endpoints\n", self.left_endpoints)

        # print("csum\n", self.csum)

    def sample_angle(self):
        u = self.csum[:, -1] * torch.rand(self.right_endpoints.size(-2))
        # print(u)

        idx = torch.searchsorted(self.csum, u.unsqueeze(-1)).squeeze(-1)
        is_gt_zero = idx > 0

        u[is_gt_zero] -= self.csum[is_gt_zero, idx[is_gt_zero] - 1]

        return self.left_endpoints[self.one_to_batch, idx] + u 

    def intersection_angles(self, p, q, r):
        """
        Solve the trigonometry inequalities
            p * cos(theta) + q * sin(theta) + r <= 0.

        Args:
            p (tensor): of size (batch, m)
            q (tensor): of size (batch, m)
            r (tensor): broadcastable with p and q
        """
        normalize = torch.sqrt(p ** 2 + q ** 2)

        if normalize.abs().lt(1e-10).any():
            raise RuntimeError

        # It's impossible that the ratio < -1 if A @ x + b <= 0.
        assert r.neg().div(normalize).ge(-1).all()

        has_solution = r.neg().div(normalize).lt(1.)

        arccos = torch.arccos(-1. * r / normalize)
        arctan = torch.arctan(q / (normalize + p))

        theta1 = -1. * arccos + 2. * arctan
        theta2 = +1. * arccos + 2. * arctan

        # theta1[~has_solution] = math.pi
        # theta2[~has_solution] = math.pi
        theta1 = theta1[:, has_solution.squeeze()]
        theta2 = theta2[:, has_solution.squeeze()]

        # translate every angle to [0, 2 * pi]
        theta1 = theta1 + theta1.lt(0.) * 2. * math.pi
        theta2 = theta2 + theta2.lt(0.) * 2. * math.pi

        theta1, theta2 = torch.minimum(theta1, theta2), torch.maximum(theta1, theta2)

        return theta1, theta2

    def candidate_endpoints(self, angles):
        """
        Construct endpoints of active elliptical slices from intersection angles.

        Args:
            angles (tuple): angles[0] and angles[1] are intersection angles of size (batch, m)

        Return:
            candidate endpoints
        """
        theta1, theta2 = angles

        srted, indices = torch.sort(theta1, descending=False)
        cummax = theta2[self.one_to_batch.unsqueeze(-1), indices].cummax(dim=-1).values

        return (
            torch.cat([srted, srted.new_ones(srted.size(-2), 1) * 2 * math.pi], dim=-1),
            torch.cat([cummax.new_zeros(cummax.size(-2), 1), cummax], dim=-1),
        )


class TruncatedGaussianSampler(object):
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def next(self, x):
        z = torch.randn(x.size(-1))

        sampler = EllipticalSliceSampler(self.A, self.b, x, z)

        theta = sampler.sample_angle()
        return x * torch.cos(theta).unsqueeze(-1) + z * torch.sin(theta).unsqueeze(-1)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # domain is -1 <= x <= 3
    A = torch.tensor([[-1.], [1.]])
    b = torch.tensor([-1., -3.])

    x = torch.tensor([0.])
    # x = torch.zeros(1, 1)

    sampler = TruncatedGaussianSampler(A, b)

    for i in range(1000):
        x = sampler.next(x)
    
    samples = []
    for i in range(50):
        for j in range(100):
            print("=== iter {:d} ===".format(j))
            x = sampler.next(x)
        samples.append(x)


    # samples = x
    samples = torch.cat(samples, dim=-2)
    mean = samples.mean(dim=-2)
    std = samples.var(dim=-2)

    print(mean, std)

    from scipy.stats import truncnorm
    mean, var = truncnorm.stats(-1, 3)
    print(mean, var)

    # import matplotlib.pyplot as plt
    # plt.hist(samples.tolist(), density=True, bins=20)
    # # plt.hist(samples.tolist(), cumulative=True, density=True, bins=20)

    # xx = np.linspace(-1, 3.)
    # # plt.plot(xx, truncnorm.cdf(xx, -1, 3), label='truncated Gaussian CDF')
    # plt.plot(xx, truncnorm.pdf(xx, -1, 3), label='truncated Gaussian PDF')
    # plt.legend()

    # # plt.title('cumulative histogram of MCMC samples')
    # plt.title('density histogram of MCMC samples')
    # plt.show()
