import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from rsome import ro
from rsome import grb_solver as grb
import rsome as rso
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import sbibm

import os
import pickle
import io
import argparse

# Maybe see how all of this changes as alpha changes
# That's why I have this function to take a list of alpha values
def list_of_ints(arg):
    return list(map(float, arg.split(',')))

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='two_moons', type=str,
                    help='The task name from the SBI benchmarks')
parser.add_argument('--max_k', default= 50, type=int,
                    help='maximum number of balls')
parser.add_argument('--alpha', default= 0.05, type=float,
                    help='alpha for conformal quantile')
parser.add_argument('--N_cal', default=1000, type=int,
                    help='number of calibration samples')
parser.add_argument('--N_test', default=1000, type=int,
                    help='number of test samples')
parser.add_argument('--samples_per_ball', default=100, type=int,
                    help='number of samples per ball')
# parser.add_argument('--list_of_alphas', default= 0.05, type=list_of_ints,
#                     help='list of alphas for conformal quantile')

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

device = ("cuda" if torch.cuda.is_available() else "cpu")

volume_d_dict = {
    2: lambda r : np.pi * r**2,
    3: lambda r : 4/3 * np.pi * r**3,
    4: lambda r : np.pi**2/2 * r**4,
    5: lambda r : 8/15 * np.pi**2 * r**5,
    6: lambda r : np.pi**3/6 * r**6,
    7: lambda r : 16/105 * np.pi**3 * r**7,
    8: lambda r : np.pi**4/24 * r**8,
    9: lambda r : 32/945 * np.pi**4 * r**9,
    10: lambda r : np.pi**5/120 * r**10
}

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def get_data(prior, simulator, N_cal = 1000, N_test = 100):
    # used for setting up dimensions
    sample_c = prior.sample((1,))
    sample_x = simulator(sample_c)

    c_dataset = prior.sample((N_cal,))
    x_dataset = simulator(c_dataset)

    to_tensor = lambda r : torch.tensor(r).to(torch.float32).to(device)
    x_cal, c_cal = to_tensor(x_dataset), to_tensor(c_dataset)

    c_test = prior.sample((N_test,))
    x_test = simulator(c_test)
    x_test, c_test = to_tensor(x_test), to_tensor(c_test)

    return (x_cal, x_test), (c_cal, c_test)


# Loads the model
def model_loader(model_path, device):
    with open(model_path, "rb") as f:
        model = CPU_Unpickler(f).load().to(device)
    return model


# Gets the conformal quantile for a given alpha and k
def conformal_quantile(alpha, k, x_cal, c_cal, encoder):
    n_c = c_cal.shape[0]
    c_cal_hat = encoder.sample(k, x_cal).detach().cpu().numpy()
    c_true = c_cal.detach().cpu().numpy().reshape(n_c, 1, -1)
    c_true = np.repeat(c_true, k, axis=1)
    scores_i = np.linalg.norm(c_cal_hat - c_true, axis=-1)
    scores = np.min(scores_i, axis=1)
    return np.quantile(scores, (n_c + 1)*(1 - alpha)/n_c)

def volume_d_ball(d, r):
    if d > 10:
        return r^d
    else:
        return volume_d_dict[d](r)

# Picks radius randomly from the interval [0, 1] and picks a direction uniformly at random
def mullers_sample_from_ball(center, r, N, d):
    u = np.random.normal(0, 1, (N, d))
    norm = np.linalg.norm(u, axis=1)
    radius = np.random.uniform(0, 1, N)**(1/d)
    u = r * radius.reshape(-1, 1) * u / norm.reshape(-1, 1)
    u = u + center 
    return u

def sample_from_k_balls_assign_voronoi(r, N, d, k, encoder, x_test):

    # Getting the centers of the k balls
    centers = encoder.sample(k, x_test).detach().cpu().numpy().reshape(k, -1)

    # Samples from the k balls
    samples = np.zeros((k, N, d))

    # Giving the voronoi assignments
    voronoi_assignments = np.zeros((k, N))

    # fraction of samples in each ball in the associated voronoi cell
    fractions = np.zeros(k)

    # vectorized 
    samples = np.apply_along_axis(mullers_sample_from_ball, axis=1, arr=centers, r=r, N=N, d=d)
    dist = np.linalg.norm(samples.reshape(k, N, 1, -1) - centers, axis=-1)
    voronoi_assignments = np.argmin(dist, axis=-1)
    fractions = np.sum(voronoi_assignments == np.arange(k).reshape(-1, 1), axis=1)/N
    volume = sum(fractions)*volume_d_ball(d, r)
    

    return centers, samples, voronoi_assignments, fractions, volume


def sample_from_k_balls_assign_voronoi_2(r, N, d, k, encoder, x_test):

    # Getting the centers of the k balls
    centers = encoder.sample(k, x_test).detach().cpu().numpy().reshape(k, -1)
    center_tiled = np.repeat(centers.reshape(k, 1, -1), N, axis=1)

    # Samples from the k balls
    samples = np.zeros((k, N, d))

    # Giving the voronoi assignments
    voronoi_assignments = np.zeros((k, N))

    # fraction of samples in each ball in the associated voronoi cell
    fractions = np.zeros(k)

    for i in range(k):
        # Sampling from the ith ball
        samples[i] = mullers_sample_from_ball(centers[i], r, N, d)

        # Giving voronoi assignments - assigning each sample to the closest center
        dist = np.linalg.norm(samples[i] - center_tiled, axis=-1)
        voronoi_assignments[i] = np.argmin(dist, axis=0)

        # Fraction of samples in the ith ball in the associated voronoi cell
        fractions[i] = np.sum(voronoi_assignments[i] == i)/N
    
    # Volume of the union of the k balls
    volume = sum(fractions)*volume_d_ball(d, r)
    

    return centers, samples, voronoi_assignments, fractions, volume

# Gridding up the space. Need to fix this

def get_grid(size = 200, dim = 2, prior= None):
    mins = prior.support.base_constraint.lower_bound.cpu().numpy()
    maxs = prior.support.base_constraint.upper_bound.cpu().numpy()
    ranges = [np.arange(mins[i], maxs[i], (maxs[i] - mins[i]) / size) for i in range(d)]
    return np.array(np.meshgrid(*ranges)).T.astype(np.float32).reshape(-1, dim)


# If dimensions are low we can compare the random estimates with the actual volume
def compute_actual_volume(radius, centers, prior):
    grid = get_grid(prior=prior)
    return grid.shape



def multiple_experiment_run(r, N, d, range_k, encoder, x_test):
    size_of_x = x_test.shape[0]
    print(len(range_k))
    volumes = np.zeros((size_of_x, len(range_k)))
    for j in range(size_of_x):
        z = 0
        print("Sample: {}/{}".format(j+ 1, size_of_x))
        for i in range_k:
            centers, _, _, fractions, volume = sample_from_k_balls_assign_voronoi(r[z], N, d, i, encoder, x_test[j].reshape(1, -1))
            volumes[j, z] = volume
            z += 1
    return volumes

        
if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    task_name = args.task_name
    max_k = args.max_k
    N_test = args.N_test
    samples_per_ball = args.samples_per_ball
    N_cal = args.N_cal
    alpha = args.alpha

    # Setting the random seed
    np.random.seed(0)

    # Sanity check
    print("Task: {}".format(task_name))

    # Getting the simulator and prior
    task = sbibm.get_task(task_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    # Getting calibration and test data
    (x_cal, x_test), (c_cal, c_test) = get_data(prior=prior, simulator=simulator, N_cal=N_cal, N_test=N_test)
    d = c_cal.shape[-1]

    print("Dimensions of Response variable: {}".format(d))

    # Loading the nf model
    cached_fn = os.path.join("../sbi/trained", f"{task_name}.nf")
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = model_loader(cached_fn, device)


    # Getting the conformal quantiles for each k
    range_k = range(1, max_k + 1)
    q_hats = [conformal_quantile(0.05, k, x_cal, c_cal, encoder) for k in range_k]

    # Getting volumes for each k of each test sample
    list_of_volumes = []
    list_of_avg_volumes = []
    volumes = multiple_experiment_run(q_hats, samples_per_ball, d, range_k, encoder, x_test)
    # for i in range(0, 10):
    #     volumes = multiple_experiment_run(q_hats, samples_per_ball, d, range_k, encoder, x_test)
    #     list_of_volumes.append(volumes)
    #     list_of_avg_volumes.append(np.mean(volumes, axis=0))
    
    # list_of_avg_volumes = np.array(list_of_avg_volumes)
    # print("Shape of list avg volume is {}".format(list_of_avg_volumes.shape))
    # Making a directory to store the results
    result_dir = os.path.join("results", task_name)
    os.makedirs(result_dir, exist_ok=True)


    # Making plot for average volume
    
    avg_volume = np.mean(volumes, axis=0)
    std_volume = np.std(volumes, axis=0)
    some = np.arange(1, max_k + 1)
    plt.figure()
    plt.plot(some, avg_volume)
    plt.errorbar(some, avg_volume, yerr=std_volume, fmt='o')
    plt.xlabel("k")
    plt.ylabel("Average Volume across {} test samples".format(N_test))
    plt.title(task_name)
    plt.show()
    plt.savefig(os.path.join("results", task_name, "avg_volume_plot.png"))

    # Making plot for average volume with error bars
    # avg_avg_volume = np.mean(list_of_avg_volumes, axis=0)
    # avg_std_volume = np.std(list_of_avg_volumes, axis=0)
    # volume_err_min = avg_avg_volume - avg_std_volume
    # volume_err_max = avg_avg_volume + avg_std_volume
    # volume_err = [volume_err_min, volume_err_max]
    # print("standard deviations are {}".format(avg_std_volume))
    # plt.figure()
    # plt.plot(some, avg_avg_volume)
    # plt.errorbar(some, avg_avg_volume, yerr=avg_std_volume, fmt='o')
    # plt.xlabel("k")
    # plt.ylabel("Average Volume across {} test samples".format(N_test))
    # plt.title(task_name)
    # plt.show()
    # plt.savefig(os.path.join("results", task_name, "avg_avg_volume_plot.png"))

    # Making plot for k/qhat
    plt.figure()
    plt.plot(some, q_hats)
    plt.xlabel("k")
    plt.ylabel("Score quantile using {} calib samples".format(N_cal))
    plt.title(task_name)
    plt.show()
    plt.savefig(os.path.join("results", task_name, "k_qhat_plot.png"))
    
    # Saving the results
    quantiles = {"k": range_k, r"$\hat{q}_k$": q_hats}
    quantiles.update({r"$\hat{q}_k$": q_hats})
    quantile_df = pd.DataFrame(quantiles)
    quantile_df.to_csv(os.path.join(result_dir, "quantiles.csv"))

    volumes_df = pd.DataFrame(volumes)
    volumes_df.to_csv(os.path.join(result_dir, "volumes.csv"))

    # avg_volumes_df = pd.DataFrame(np.hstack([some.reshape(-1, 1), avg_avg_volume.reshape(-1, 1), avg_std_volume]), columns=["k", "avg_volume", "std_volume"])
    # avg_volumes_df.to_csv(os.path.join(result_dir, "avg_volumes.csv"))
    
    with open(os.path.join(result_dir, "volumes.npy"), 'wb') as f:
        np.save(f, volumes)

