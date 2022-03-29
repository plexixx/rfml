"""
Substitute Model for Adversarial Training.
    Reference:
        Alireza Bahramali, Milad Nasr, Amir Houmansadr, Dennis Goeckel, Don Towsley
        "Robust Adversarial Attacks Against DNN-Based Wireless Communication Systems"
        https://arxiv.org/abs/2102.00918
"""
__author__ = "plexix <byx1029@bupt.edu.cn>"

# External Includes
import torch.nn as nn

# Internal Includes
from .base import Model
from rfml.nn.layers import Flatten, PowerNormalization

class Substitute(Model):
    def __init__(self, input_samples: int, n_classes: int):
        super().__init__(input_samples, n_classes)

        self.preprocess = PowerNormalization()

        self.dense1 = nn.Linear(1, 1024)
        self.a1 = nn.ReLU()
        self.n1 = nn.BatchNorm1d(1024)
        
        self.dense2 = nn.Linear(1024, 1024)
        self.a2 = nn.ReLU()
        self.n2 = nn.BatchNorm1d(1024)

        self.dense3 = nn.Linear(1024, 512)
        self.a3 = nn.ReLU()
        self.n3 = nn.BatchNorm1d(512)

        self.dense4 = nn.Linear(512, 128)
        self.a4 = nn.ReLU()
        self.n4 = nn.BatchNorm1d(128)

        self.dense5 = nn.Linear(128, n_classes)
        self.a5 = nn.Softmax()


    def forward(self, x):
        x = self.preprocess(x)

        x = self.dense1(x)
        x = self.a1(x)
        x = self.n1(x)

        x = self.dense2(x)
        x = self.a2(x)
        x = self.n2(x)

        x = self.dense3(x)
        x = self.a3(x)
        x = self.n3(x)

        x = self.dense4(x)
        x = self.a4(x)
        x = self.n4(x)

        x = self.dense5(x)
        x = self.a5(x)

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