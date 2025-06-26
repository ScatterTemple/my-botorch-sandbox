import numpy as np
from numpy import sin
import torch
import plotly.graph_objs as go

# approximate
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll

# exact
from botorch.models.fully_bayesian import FullyBayesianSingleTaskGP
from botorch.fit import fit_fully_bayesian_model_nuts


def objective(x):
    return sin(x)


def gen_train_x(dim, n):
    return torch.tensor(np.random.rand(n, dim))


def plot_scatter_1d(x, y, fig=None):
    if fig is None:
        fig = go.Figure()

    fig.scatter(x=x, y=y, mode='markers',)

    return fig



