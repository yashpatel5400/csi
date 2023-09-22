import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rsome import ro 
import rsome as rso                           # import the ro module
from rsome import grb_solver as grb
from torch.utils.data import DataLoader, TensorDataset
import sbibm
import model


device = ("cuda" if torch.cuda.is_available() else "cpu")


# The nodes are in row-major order
def make_grid(n_rows, n_cols):
    vertex_count = n_rows * n_cols
    edges = np.zeros((vertex_count, vertex_count))
    edge_list = {}
    adjacency_list = {i : [[], []] for i in range(vertex_count)}
    grid_size = 5

    edge_index = 0
    for i in range(grid_size): # row
        for j in range(grid_size): # column
            if i <(n_rows -1) and j < (n_cols -1):
                # East edge
                edges[i * n_cols + j, i * n_cols + j + 1] = 1
                edge_list[(i * n_cols + j, i * n_cols + j + 1)] = edge_index
                edge_index += 1
                adjacency_list[i * n_cols + j][0].append(i *n_cols + j + 1)
                adjacency_list[i * n_cols + j + 1][1].append(i * n_cols + j)

                # South edge
                edges[i * n_cols + j, (i + 1) * n_cols + j] = 1
                edge_list[(i * n_cols + j, (i + 1) * n_cols + j)] = edge_index
                edge_index += 1
                adjacency_list[i * n_cols + j][0].append((n_cols + 1) * 5 + j)
                adjacency_list[(i + 1) * n_cols + j][1].append(n_cols * 5 + j)
            elif i== (n_rows -1) and j < (n_cols -1):
                # East edge
                edges[i * n_cols + j, i * n_cols + j + 1] = 1
                edge_list[(i * n_cols + j, i * n_cols + j + 1)] = edge_index
                edge_index += 1
                adjacency_list[i * n_cols + j][0].append(i * n_cols + j + 1)
                adjacency_list[i * 5 + n_cols + 1][1].append(i * n_cols + j)

            elif i < (n_rows -1) and j == (n_cols -1):
                # South edge
                edges[i * n_cols + j, (i + 1) * n_cols + j] = 1
                edge_list[(i * n_cols + j, (i + 1) * n_cols + j)] = edge_index
                edge_index += 1
                adjacency_list[i * n_cols + j][0].append((n_cols + 1) * 5 + j)
                adjacency_list[(i + 1) * n_cols + j][1].append(i * n_cols + j)
    
    edge_count = int(edges.sum())
    rev_edge_list = {v : k for k, v in edge_list.items()}
    return edges, edge_list, rev_edge_list, adjacency_list, vertex_count, edge_count



def generate_z_costs(N, d, Theta, edge_count):
    # Generating zs
    z = np.random.normal(0, 1, (N, d))

    # Computing cost with some noise
    cost_wo_noise = ((1/d)**(0.5) * np.matmul(z, Theta.T) + 3)**5 + 1
    noise = np.random.uniform(low = 0.75, high = 1.25, size = (N, edge_count))
    cost = cost_wo_noise * noise
    return z, cost 

# Mapping the zs that come from the generator. I will be conformalizing in the C = f(Z) space 
def f_z(z, Theta):
    d = z.shape[1]
    return ((1/d)**(0.5) * np.matmul(z, Theta.T) + 3)**5 + 1


def get_data(prior, simulator, N_cal = 1000, N_test = 100, mapping = f_z, *argv):
    # used for setting up dimensions
    sample_z = prior.sample((1,))
    sample_x = simulator(sample_z)

    z_dataset = prior.sample((N_cal,))
    x_dataset = simulator(z_dataset)
    c_dataset = mapping(z_dataset, argv[0])

    to_tensor = lambda r : torch.tensor(r).to(torch.float32).to(device)
    x_cal, c_cal = to_tensor(x_dataset), to_tensor(c_dataset)

    z_test = prior.sample((N_test,))
    x_test = simulator(z_test)
    c_test = mapping(z_test, argv[0])
    x_test, c_test = to_tensor(x_test), to_tensor(c_test)

    return (x_cal, x_test), (c_cal, c_test)

# Gets the conformal quantile for a given alpha and k
def conformal_quantile(alpha, k, x_cal, c_cal, encoder, mapping, *argv):
    n_c = c_cal.shape[0]
    z_cal_hat = encoder.sample(k, x_cal).detach().cpu().numpy()
    c_cal_hat = mapping(z_cal_hat, argv[0])

    c_true = c_cal.detach().cpu().numpy().reshape(n_c, 1, -1)
    c_true = np.repeat(c_true, k, axis=1)
    scores_i = np.linalg.norm(c_cal_hat - c_true, axis=-1)
    scores = np.min(scores_i, axis=1)
    return np.quantile(scores, (n_c + 1)*(1 - alpha)/n_c)
