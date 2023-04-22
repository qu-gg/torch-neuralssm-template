"""
@file layers.py

Miscellaneous helper Torch layers
"""
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        """
        Handles flattening a Tensor within a nn.Sequential Block

        :param input: Torch object to flatten
        """
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, w):
        """
        Handles unflattening a vector into a 4D vector in a nn.Sequential Block

        :param w: width of the unflattened image vector
        """
        super().__init__()
        self.w = w

    def forward(self, input):
        nc = input[0].numel() // (self.w ** 2)
        return input.view(input.size(0), nc, self.w, self.w)
