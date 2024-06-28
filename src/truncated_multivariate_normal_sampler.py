import torch
import warnings

from . import EllipticalSliceSampler


class TruncatedMultivariateNormalSampler(object):
    def __init__(self, mean, covariance, A, b):
        self.mean = mean
        self.covariance = covariance

        self.A = A
        self.b = b

    def draw(self):
        sampler = EllipticalSliceSampler(self.A, self.b)

        new_x = sampler.sample_slice()

        return new_x
