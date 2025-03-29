import torch
from ess import EllipticalSliceSampler


class TruncatedMultivariateNormal:
    def __init__(self, mean, covariance_matrix, A, b):
        self.mean = mean
        self.covariance_matrix = covariance_matrix

        self.covar_root = torch.linalg.cholesky(covariance_matrix)

        self.A = A
        self.b = b

    def sample(self, num_steps, burnin=100, thinning=10):
        sampler = EllipticalSliceSampler(
            A=self.A @ self.covar_root,
            b=self.b - self.A @ self.mean,
            burnin=burnin,
        )

        samples = sampler.launch(num_steps=num_steps, thinning=thinning)

        return self.mean + samples @ self.covar_root.T
