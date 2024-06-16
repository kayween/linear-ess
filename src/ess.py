import math
import torch

import warnings


class EllipticalSliceSampler(object):
    def __init__(self, A, b, x, burnin=0):
        """
        Elliptical slice sampler for multivariate standard Gaussian distributions under
        linear inequality constraints A @ x <= b.

        Args:
            A (tensor): The matrix of size (m, d) in the linear inequality constraints A @ x <= b.
            b (tensor): The vector of size (d,) in the constraints.
            x (tensor): The initialization of size (*, d). Has to satisfy all linear inequality constraints.
                Its first dimension is the batch size, which will be used as the number of independent Markov chains.
        """
        if (x @ A.T - b).gt(0.).any():
            raise RuntimeError

        self.A = A
        self.b = b

        self.x = x = x.unsqueeze(-2) if x.dim() == 1 else x

        # Some common tensors that will be used later
        batch = x.size(-2)

        self.zeros = x.new_zeros((batch, 1))
        self.ones = x.new_ones((batch, 1))
        self.indices_batch = torch.arange(batch, dtype=torch.int64, device=x.device)

        # Count the total number of fallback.
        self.cnt_violations = 0
        self.total_iterations = 0

        # Let's burn some samples.
        self.launch(burnin)

    def trim_active_angles(self, left, right):
        """
        Trim the active slices (by at most 1e-6) to ensure the samples stay inside the domain.
        """
        gap = torch.clamp(right - left, min=0.)
        eps = gap.mul(0.25).clamp(max=1e-6)

        return left + eps, right - eps

    def draw_angles(self, left, right):
        csum = right.sub(left).clamp(min=0.).cumsum(dim=-1)

        u = csum[:, -1] * torch.rand(right.size(-2), device=self.x.device)

        # The returned index i satisfies csum[i - 1] < u <= csum[i]
        idx = torch.searchsorted(csum, u.unsqueeze(-1)).squeeze(-1)

        # Do a zero padding so that padded_csum[i] = csum[i - 1]
        padded_csum = torch.cat([self.zeros, csum], dim=-1)

        return u - padded_csum[self.indices_batch, idx] + left[self.indices_batch, idx]

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

        if radius.abs().lt(1e-6).any():
            warnings.warn("The ellipse has an extremely small volume. This may cause numerical issues.")

        # It's impossible that the ratio < -1 if A @ x <= b.
        assert bias.div(radius).ge(-1).all()

        if bias.div(radius).min().le(-1 + 1e-6):
            warnings.warn("The ellipse is almost outside the domain. This may cause numerical issues.")

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

    def step(self, x):
        """
        Do one step elliptical slice sampling. Fall back to previous state if the samples violate the constraint.
        Each iteration has three matrix-vector multiplications. Can be reduced to two if caching A @ x.
        """
        self.total_iterations += 1

        nu = torch.randn(x.size(), dtype=x.dtype, device=x.device)

        alpha, beta = self.intersection_angles(x @ self.A.T, nu @ self.A.T, self.b.unsqueeze(-2))
        left, right = self.active_angles(alpha, beta)
        left, right = self.trim_active_angles(left, right)

        theta = self.draw_angles(left, right).unsqueeze(-1)
        candidate = x * torch.cos(theta) + nu * torch.sin(theta)

        is_feasible = (candidate @ self.A.T - self.b).lt(0.).all(dim=-1)
        x[is_feasible] = candidate[is_feasible]

        if not is_feasible.all() > 0:
            warnings.warn("Some Markov chains fall back to previous state due to constraint violations.")
            self.cnt_violations += torch.sum(~is_feasible)

        return x

    def launch(self, num_steps, thinning=None):
        samples = []

        for i in range(num_steps):
            self.x = self.step(self.x)

            if thinning is not None and i % thinning == 0:
                samples.append(self.x.clone().detach())

        if thinning is not None:
            return torch.cat(samples, dim=-2)
        else:
            return self.x
