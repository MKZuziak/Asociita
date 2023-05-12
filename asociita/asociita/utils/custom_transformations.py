from typing import Any
import torch.random


class AddGaussianNoise(object):
    def __init__(self, 
                 mean=0., 
                 std=1.,
                 noise_multiplication: float = 0.5) -> None:
        self.std = std
        self.mean = mean
        self.noise_multiplication = noise_multiplication
    

    def __call__(self, 
                 tensor) -> Any:
        return tensor + self.noise_multiplication * (torch.randn(tensor.size()) * self.std + self.mean)
    

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)