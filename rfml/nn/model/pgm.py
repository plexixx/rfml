"""
Perturbation Generator Model.
    Reference:
        Alireza Bahramali, Milad Nasr, Amir Houmansadr, Dennis Goeckel, Don Towsley
        "Robust Adversarial Attacks Against DNN-Based Wireless Communication Systems"
        https://arxiv.org/abs/2102.00918
"""
__author__ = "plexix <byx1029@bupt.edu.cn>"

# External Includes
import torch.nn as nn
import torch.optim as optim

# Internal Includes
from .base import Model
from rfml.nn.layers import Flatten, PowerNormalization

class PGM(Model):
    def __init__(self, input_samples: int, n_classes: int):
        super().__init__(input_samples, n_classes)

        self.preprocess = PowerNormalization()

        self.dense1 = nn.Linear(1, 5000)
        self.a1 = nn.LeakyReLU()
        self.n1 = nn.BatchNorm1d(5000)

        self.dense2 = nn.Linear(5000, 1000)
        self.a2 = nn.LeakyReLU()
        self.n2 = nn.BatchNorm1d(1000)

        self.dense3 = nn.Linear(1000, n_classes)
    

    def forward(self, x):
        x = self.preprocess(x)

        x = self.dense1(x)
        x = self.a1(x)
        x = self.n1(x)

        x = self.dense2(x)
        x = self.a2(x)
        x = self.n2(x)

        x = self.dense3(x)

        return x
    """
    def _freeze(self):
        \"""Freeze all of the parameters except for the dense layers.
        \"""
        for name, module in self.named_children():
            if "dense" not in name and "n3" not in name and "n4" not in name:
                for p in module.parameters():
                    p.requires_grad = False

    def _unfreeze(self):
        \"""Re-enable training of all parameters in the network.
        \"""
        for p in self.parameters():
            p.requires_grad = True
    """