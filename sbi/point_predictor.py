"""
To launch all the tasks, create tmux sessions (separately for each of the following) 
and run (for instance):

python canvi_sbibm.py --task two_moons --cuda_idx 0
python canvi_sbibm.py --task slcp --cuda_idx 1
python canvi_sbibm.py --task gaussian_linear_uniform --cuda_idx 2
python canvi_sbibm.py --task bernoulli_glm --cuda_idx 3
python canvi_sbibm.py --task gaussian_mixture --cuda_idx 4
python canvi_sbibm.py --task gaussian_linear --cuda_idx 5
python canvi_sbibm.py --task slcp_distractors --cuda_idx 6
python canvi_sbibm.py --task bernoulli_glm_raw --cuda_idx 7
"""

import pandas as pd
import numpy as np
import sbibm
import torch
import math
import torch.distributions as D
import matplotlib.pyplot as plt

from pyknos.nflows import flows, transforms
from functools import partial
from typing import Optional
from warnings import warn

from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows, transforms
from pyknos.nflows.nn import nets
from pyknos.nflows.transforms.splines import rational_quadratic
from torch import Tensor, nn, relu, tanh, tensor, uint8

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

sns.set_theme()

from sbi.utils.sbiutils import (
    standardizing_net,
    standardizing_transform,
    z_score_parser,
)
from sbi.utils.torchutils import create_alternating_binary_mask
from sbi.utils.user_input_checks import check_data_device, check_embedding_net_device

import os
import pickle
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, sample_x, sample_y):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(sample_x.shape[-1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, sample_y.shape[-1])
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        rep = F.relu(self.fc3(x)) # used as "learned representation" for conditional generation
        x = self.fc4(rep)
        return x, rep

def train_model():
    task_name = "two_moons"
    task = sbibm.get_task(task_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    sample_y = prior.sample((1,))
    sample_x = simulator(sample_y)

    model = SimpleModel(sample_x, sample_y)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    n_epochs = 1_000
    sims_per_epoch = 100

    losses = []
    for epoch in range(n_epochs):
        y = prior.sample((sims_per_epoch,))
        x = simulator(y)

        y_pred, _ = model(x)
        loss = loss_fn(y_pred, y)
        losses.append(loss.detach().numpy())
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"epoch {epoch} loss {loss}")

    torch.save(model, f"{task_name}.pt")

if __name__ == "__main__":
    train_model()