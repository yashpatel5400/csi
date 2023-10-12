import numpy as np
import matplotlib.pyplot as plt
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
import argparse

import os
import pickle
import sys, os
import multiprocessing

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(FeedforwardNN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

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

def get_data(prior, simulator):
    # used for setting up dimensions
    sample_c = prior.sample((1,))
    sample_x = simulator(sample_c)

    N = 2_000
    N_train = 1000
    N_test  = 1000

    c_dataset = prior.sample((N,))
    x_dataset = simulator(c_dataset)

    to_tensor = lambda r : torch.tensor(r).to(torch.float32).to(device)
    x_train, x_cal = to_tensor(x_dataset[:N_train]), to_tensor(x_dataset[N_train:])
    c_train, c_cal = to_tensor(c_dataset[:N_train]), to_tensor(c_dataset[N_train:])

    c_test = prior.sample((N_test,))
    x_test = simulator(c_test)
    x_test, c_test = to_tensor(x_test), to_tensor(c_test)

    return (x_train, x_cal, x_test), (c_train, c_cal, c_test)


def get_point_predictor(x_train, c_train):
    train_dataset = TensorDataset(x_train, c_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = FeedforwardNN(input_dim=x_train.shape[-1], output_dim=c_train.shape[-1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1_000

    for epoch in range(num_epochs):
        for batch_X, batch_c in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch_X)  # Forward pass
            loss = criterion(outputs, batch_c)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model


# *marginal* box constraint (i.e. just ignore contextual information)
def nominal_solve(generative_model, alpha, x, c_true, p, B):
    # perform RO over constraint region
    
    # leaving this here if anyone is curious to see that vectorizing linear optimization is equivalent
    # ind_values = []
    # for i in range(len(c_true)):
    #     model = ro.Model()

    #     w = model.dvar(c_true[i].shape)
    #     c_i = c_true.detach().cpu().numpy()[i]
        
    #     model.min(-(c_i * w).sum())
    #     model.st(w <= 1)
    #     model.st(w >= 0)
    #     model.st(p[i] @ w <= B[i])
        
    #     blockPrint()
    #     model.solve(grb)
    #     enablePrint()
    #     ind_values.append(model.get())

    model = ro.Model()

    w = model.dvar(c_true.shape)
    c = c_true.detach().cpu().numpy()
    
    model.min(-(c * w).sum())
    model.st(w <= 1)
    model.st(w >= 0)
    model.st(p[i] @ w[i,:] <= B[i] for i in range(len(B)))
    
    blockPrint()
    model.solve(grb)
    enablePrint()
    
    w_star = w.get()
    optima = np.sum(-(c * w_star), axis=-1)
    return 1, optima # model.get()


def box_solve_generic(c_box_lb, c_box_ub, c_true, p, B):
    # c_true_np = c_true.detach().cpu().numpy()
    # covered = int(np.all(c_box_lb <= c_true_np) and np.all(c_true_np <= c_box_ub))
    covered = 0
    
    # perform RO over constraint region
    model = ro.Model()

    w = model.dvar(c_true.shape)
    c = model.rvar(c_true.shape)
    uset = (c_box_lb <= c, c <= c_box_ub)

    model.minmax(-(c * w).sum(), uset)
    model.st(w <= 1)
    model.st(w >= 0)
    # model.st(p @ w <= B)
    model.st(p[i] @ w[i,:] <= B[i] for i in range(len(B)))
    blockPrint()
    model.solve(grb)
    enablePrint()

    w_star = w.get()
    
    # perform RO over constraint region
    model = ro.Model()
    c = model.dvar(c_true.shape)
    
    model.max(-(c * w_star).sum())
    model.st(c_box_lb <= c[i] for i in range(len(B)))
    model.st(c[i] <= c_box_ub for i in range(len(B)))
    blockPrint()
    model.solve(grb)
    enablePrint()

    c_star = c.get()
    
    # calculate per trial opt
    optima = np.sum(-(c_star * w_star), axis=-1)
    return covered, optima


# *marginal* box constraint (i.e. just ignore contextual information)
def box_solve_marg(generative_model, alpha, x, c_true, p, B):
    alpha = alpha / c_true.shape[-1] # Bonferroni
    c_box_lb = np.quantile(c_dataset, q=(alpha / 2), axis=0)
    c_box_ub = np.quantile(c_dataset, q=(1 - alpha / 2), axis=0)
    return box_solve_generic(c_box_lb, c_box_ub, c_true, p, B)


# *conditional* box conformal constraint (PTC-B)
def box_solve_cp(generative_model, alpha, x, c_true, p, B):
    c_pred = generative_model.sample(1, x_cal).squeeze()
    box_cal_scores = np.linalg.norm((c_pred - c_cal).detach().cpu().numpy(), np.inf, axis=1)
    conformal_quantile = np.quantile(box_cal_scores, q=1 - alpha, axis=0)

    c_box_hat = generative_model.sample(1, x).squeeze().detach().cpu().numpy()
    c_box_lb = c_box_hat - conformal_quantile
    c_box_ub = c_box_hat + conformal_quantile
    return box_solve_generic(c_box_lb, c_box_ub, c_true, p, B)


def ellipsoid_solve_generic(c_ellipse_center, c_ellipse_axes, c_ellipse_cutoff, c_true, p, B):
    # perform RO over constraint region
    model = ro.Model()

    w = model.dvar(c_true.shape[-1])
    c = model.rvar(c_true.shape[-1])

    uset = rso.norm((c - c_ellipse_center).T @ c_ellipse_axes, 2) <= c_ellipse_cutoff
    c_true_np = c_true.detach().cpu().numpy()
    covered = int(np.linalg.norm((c_true_np - c_ellipse_center).T @ c_ellipse_axes) <= c_ellipse_cutoff)

    model.minmax(-c @ w, uset)
    model.st(w <= 1)
    model.st(w >= 0)
    model.st(p @ w <= B)

    blockPrint()
    model.solve(grb)
    enablePrint()
    return covered, model.get()


# *marginal* ellipsoid constraint
def ellipsoid_solve_marg(generative_model, alpha, x, c_true, p, B):
    mu = np.mean(c_dataset, axis=0)
    cov = np.cov(c_dataset.T)
    sqrt_cov_inv = np.linalg.cholesky(np.linalg.inv(cov))
    mah_dists = np.linalg.norm((c_dataset - mu) @ sqrt_cov_inv, axis=1)
    cutoff = np.quantile(mah_dists, q=1 - alpha)
    
    return ellipsoid_solve_generic(mu, sqrt_cov_inv, cutoff, c_true, p, B)


# *conditional* ellipsoid conformal constraint (PTC-E)
def ellipsoid_solve_cp(generative_model, alpha, x, c_true, p, B):
    c_pred = generative_model.sample(1, x_cal).squeeze()
    residuals = (c_pred - c_cal).detach().cpu().numpy()

    cov = np.cov(residuals.T)
    sqrt_cov_inv = np.linalg.cholesky(np.linalg.inv(cov))
    ellipsoid_cal_scores = np.linalg.norm(residuals @ sqrt_cov_inv, axis=1)
    conformal_quantile = np.quantile(ellipsoid_cal_scores, q=1 - alpha, axis=0)

    c_ellipsoid_hat = generative_model.sample(1, x.unsqueeze(0)).squeeze().detach().cpu().numpy()
    return ellipsoid_solve_generic(c_ellipsoid_hat, sqrt_cov_inv, conformal_quantile, c_true, p, B)


# current f = -w^T c --> grad_w(f) = -c
def grad_f(w, c):
    return -c


def inner_opt(args):
    w, c_shape, conformal_quantile, c_region_center = args
    model = ro.Model()
    c = model.dvar(c_shape)

    model.max(-(c * w).sum())
    model.st(rso.norm(c[i] - c_region_center[i], 2) <= conformal_quantile for i in range(c_shape[0]))
    blockPrint()
    model.solve(grb)
    enablePrint()

    return c.get()#, model.get()


# generative model-based prediction regions
def cpo(generative_model, alpha, x, c_true, p, B):
    def get_quantile(k = 10):
        c_cal_hat = generative_model.sample(k, x_cal).detach().cpu().numpy()
        c_cal_tiled = np.transpose(np.tile(c_cal.detach().cpu().numpy(), (k, 1, 1)), (1, 0, 2))
        c_cal_diff = c_cal_hat - c_cal_tiled
        c_cal_norms = np.linalg.norm(c_cal_diff, axis=-1)
        c_cal_scores = np.min(c_cal_norms, axis=-1)

        return np.quantile(c_cal_scores, q = 1 - alpha)

    K = 10
    conformal_quantile = get_quantile(K)
    c_region_centers = generative_model.sample(K, x).detach().cpu().numpy()
    c_tiled = np.transpose(np.tile(c_true.detach().cpu().numpy(), (K, 1, 1)), (1, 0, 2))
    c_diff = c_region_centers - c_tiled
    c_norm = np.linalg.norm(c_diff, axis=-1)
    c_score = np.min(c_norm, axis=-1)
    
    contained = np.sum(c_score < conformal_quantile) / len(c_score)
    print(f"Contained: {contained}")
    # c_region_centers = c_region_centers[0]

    eta = 5e-3 # learning rate
    T = 2_000 # optimization steps

    pool = multiprocessing.Pool(K)

    w = np.random.random(c_true.shape) / 2
    
    # start optimization loop
    for t in range(T):
        # maxizer_per_region = []
        # opt_value = []
        # maxizer, opt = inner_opt((w, c_true.shape, conformal_quantile, c_region_centers[:,0]))
        # maxizer_per_region = np.array([maxizer])
        # opt_value = np.array([opt])

        maxizer_per_region = np.array(list(zip(*pool.map(
                                                inner_opt,
                                                [(w, c_true.shape, conformal_quantile, c_region_centers[:,k]) for k in range(K)]
                                                )
                                            ))) # batch_size x K x c_dim
        w_tiled = np.transpose(np.tile(w, (K, 1, 1)), (1, 0, 2))

        opt_value_per_region = np.sum(-(maxizer_per_region * w_tiled), axis=-1)
        opt_value = np.max(opt_value_per_region, axis=-1)        
        c_star_idx = np.argmax(opt_value_per_region, axis=-1)        
        c_star = maxizer_per_region[np.arange(maxizer_per_region.shape[0]), c_star_idx] # yuck, is this really the best way?
        
        # c_star = maxizer_per_region[np.argmax(opt_value)]
        grad = grad_f(w, c_star)
        w_temp = (w - eta * grad).flatten()

        # projection step: there's probably a better way of doing this?
        model = ro.Model()
        w_d = model.dvar(w_temp.shape)

        model.min(rso.sumsqr(w_d - w_temp))
        model.st(w_d <= 1)
        model.st(w_d >= 0)

        model.st(p[i] @ w_d[i * p.shape[-1]:(i+1) * p.shape[-1]] <= B[i] for i in range(len(B)))
        blockPrint()
        model.solve(grb)
        enablePrint()

        w = w_d.get()
        w = w.reshape(c_true.shape)

        print(f"Completed step={t} -- {opt_value}")
    pool.close()
    return contained, np.max(opt_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    args = parser.parse_args()
    task_name = args.task
    
    task = sbibm.get_task(task_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    (x_train, x_cal, x_test), (c_train, c_cal, c_test) = get_data(prior, simulator)
    c_dataset = torch.vstack([c_train, c_cal]).detach().cpu().numpy() # for cases where only marginal draws are used, no splitting occurs
    # point_predictor = get_point_predictor(x_train, c_train)

    cached_fn = os.path.join("trained", f"{task_name}.nf")
    with open(cached_fn, "rb") as f:
        generative_model = pickle.load(f)
    generative_model.to(device)

    result_dir = os.path.join("results", task_name)
    os.makedirs(result_dir, exist_ok=True)
    
    alphas = [0.05]
    name_to_method = {
        # "Nominal": nominal_solve,
        # "Box": box_solve_marg,
        # "PTC-B": box_solve_cp,
        "Ellipsoid": ellipsoid_solve_marg,
        # "PTC-E": ellipsoid_solve_cp,
        # "CPO": cpo,
    }
    method_coverages = {r"$\alpha$": alphas}
    method_values = {r"$\alpha$": alphas}
    method_std = {r"$\alpha$": alphas}

    n_trials = 5

    # want these to be consistent for comparison between methods, so generate once beforehand
    ps = np.random.randint(low=0, high=1000, size=(n_trials, c_dataset.shape[-1]))
    us = np.random.uniform(low=0, high=1, size=n_trials)
    Bs = np.random.uniform(np.max(ps, axis=1), np.sum(ps, axis=1) - us * np.max(ps, axis=1))
    
    for method_name in name_to_method:
        print(f"Running: {method_name}")
        for alpha in alphas:
            covered = 0
            values = []
            trial_runs = {}
            
            for trial_idx in range(n_trials):
                x = x_test[:n_trials]
                c = c_test[:n_trials]
                p = ps[:n_trials]
                B = Bs[:n_trials]
                
                (covered_trial, value_trial) = name_to_method[method_name](generative_model, alpha, x, c, p, B)
                covered += covered_trial
                values.append(value_trial)

                trial_runs[trial_idx] = value_trial
                trial_df = pd.DataFrame(trial_runs, index=[0])
                trial_df.to_csv(os.path.join(result_dir, f"{method_name}.csv"))

            if method_name not in method_coverages:
                method_coverages[method_name] = []
                method_values[method_name] = []
                method_std[method_name] = []

            method_coverages[method_name].append(covered / n_trials)
            method_values[method_name].append(np.mean(values))
            method_std[method_name].append(np.std(values))

    coverage_df = pd.DataFrame(method_coverages)
    values_df = pd.DataFrame(method_values)
    std_df = pd.DataFrame(method_std)

    coverage_df.to_csv(os.path.join(result_dir, "coverage.csv"))
    values_df.to_csv(os.path.join(result_dir, "values.csv"))
    std_df.to_csv(os.path.join(result_dir, "std.csv"))