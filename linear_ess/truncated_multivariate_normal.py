import torch
from .ess import EllipticalSliceSampler


class TruncatedMultivariateNormal:
    def __init__(self, A, b, loc, covariance_matrix=None, scale_tril=None):
        self.A = A
        self.b = b

        self.loc = loc

        if scale_tril is not None:
            self.scale_tril = scale_tril
        elif covariance_matrix is not None:
            self.scale_tril = torch.linalg.cholesky(covariance_matrix)
        else:
            raise RuntimeError

    def sample(self, num_steps, burnin=100, thinning=10):
        sampler = EllipticalSliceSampler(
            A=self.A @ self.scale_tril,
            b=self.b - self.A @ self.loc,
            burnin=burnin,
        )

        samples = sampler.launch(num_steps=num_steps, thinning=thinning)

        return self.loc + samples @ self.scale_tril.T
