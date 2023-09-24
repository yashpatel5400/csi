import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch

import networkx as nx
import pandas as pd
import rsome as rso
import osmnx as ox
from rsome import ro
from rsome import grb_solver as grb

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


def normed_ball_solve_generic(ball_center, ball_radius, A, b, norm):
    model = ro.Model()

    w = model.dvar(A.shape[-1])
    c = model.rvar(A.shape[-1])
    uset = rso.norm(c - ball_center, norm) <= ball_radius

    model.minmax(c @ w, uset)
    model.st(w <= 1)
    model.st(w >= 0)
    model.st(A @ w == b)

    model.solve(grb)
    return model.get()


# *marginal* box constraint (i.e. just ignore contextual information)
def normed_ball_solve_marg(cal_true_traffics, cal_pred_traffics, alpha, test_pred_traffic, A, b, norm):
    mu = np.mean(cal_true_traffics, axis=0)
    box_cal_scores = np.linalg.norm(cal_true_traffics - mu, norm, axis=1)
    conformal_quantile = np.quantile(box_cal_scores, q=1 - alpha, axis=0)
    return normed_ball_solve_generic(mu, conformal_quantile, A, b, norm)


# *conditional* box conformal constraint (PTC-B)
def normed_ball_solve_cp(cal_true_traffics, cal_pred_traffics, alpha, test_pred_traffic, A, b, norm):
    sample_idx = 0 # which index of the generative model to use as point prediction
    box_cal_scores = np.linalg.norm(cal_true_traffics - cal_pred_traffics[:,sample_idx], norm, axis=1)
    conformal_quantile = np.quantile(box_cal_scores, q=1 - alpha, axis=0)
    return normed_ball_solve_generic(test_pred_traffic[sample_idx], conformal_quantile, A, b, norm)


# current f = w^T c --> grad_w(f) = c
def grad_f(w, c):
    return c


# generative model-based prediction regions
def cpo(cal_true_traffics, cal_pred_traffics, alpha, test_pred_traffic, A, b):
    k = 10

    c_cal_tiled = np.transpose(np.tile(cal_true_traffics, (k, 1, 1)), (1, 0, 2))
    c_cal_diff = cal_pred_traffics - c_cal_tiled
    c_cal_norms = np.linalg.norm(c_cal_diff, axis=-1)
    c_cal_scores = np.min(c_cal_norms, axis=-1)

    conformal_quantile = np.quantile(c_cal_scores, q = 1 - alpha)

    eta = 5e-3 # learning rate
    T = 2_500 # optimization steps
    w = np.random.random(A.shape[-1]) / 2
    
    opt_values = []
    for t in range(T):
        maxizer_per_region = []
        opt_value = []

        for c_region_center in test_pred_traffic:
            model = ro.Model()
            c = model.dvar(A.shape[-1])

            model.max(c @ w)
            model.st(rso.norm(c - c_region_center, 2) <= conformal_quantile)
            model.solve(grb)

            maxizer_per_region.append(c.get())
            opt_value.append(model.get())

        opt_values.append(np.max(opt_value))
        c_star = maxizer_per_region[np.argmax(opt_value)]
        grad = grad_f(w, c_star)
        w_temp = w - eta * grad

        # projection step: there's probably a better way of doing this?
        model = ro.Model()
        w_d = model.dvar(A.shape[-1])

        model.min(rso.norm(w_d - w_temp, 2))
        model.st(w_d <= 1)
        model.st(w_d >= 0)
        model.st(A @ w_d == b)

        model.solve(grb)

        w = w_d.get()

        print(f"Completed step={t}")
    return np.max(opt_value)


if __name__ == "__main__":
    result_dir = os.path.join("results")
    os.makedirs(result_dir, exist_ok=True)
    
    alphas = [0.05]
    name_to_method = {
        # "Box": normed_ball_solve_marg,
        # "PTC-B": normed_ball_solve_cp,
        # "Ellipsoid": normed_ball_solve_marg,
        # "PTC-E": normed_ball_solve_cp,
        "CPO": cpo,
    }
    method_values = {r"$\alpha$": alphas}
    method_std = {r"$\alpha$": alphas}

    n_trials = 1

    # problem setup
    G = ox.graph_from_place("Manhattan, New York City, New York", network_type="drive")

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    edges = ox.graph_to_gdfs(G, nodes=False)
    edges["highway"] = edges["highway"].astype(str)
    edges.groupby("highway")[["length", "speed_kph", "travel_time"]].mean().round(1)

    hwy_speeds = {"residential": 35, "secondary": 50, "tertiary": 60}
    G = ox.add_edge_speeds(G, hwy_speeds)
    G = ox.add_edge_travel_times(G)

    A = nx.incidence_matrix(G, oriented=True).todense()
    b = np.zeros(len(G.nodes)) # b entries: 1 for source, -1 for target, 0 o.w.
    b[1]   = -1
    b[120] = 1 

    # ---
    cal_true_traffics = np.load("cal_true_traffics.npy")
    cal_pred_traffics = np.load("cal_pred_traffics.npy")
    test_pred_traffics = np.load("test_pred_traffics.npy")

    for method_name in name_to_method:
        print(f"Running: {method_name}")
        for alpha in alphas:
            values = []
            trial_runs = {}
            
            for trial_idx in range(n_trials):
                if method_name == "CPO":
                    value_trial = name_to_method[method_name](cal_true_traffics, cal_pred_traffics, alpha, test_pred_traffics[trial_idx], A, b)
                else:
                    if method_name in ["Box", "PTC-B"]:
                        value_trial = name_to_method[method_name](cal_true_traffics, cal_pred_traffics, alpha, test_pred_traffics[trial_idx], A, b, norm=np.inf)
                    elif method_name in ["Ellipsoid", "PTC-E"]:
                        value_trial = name_to_method[method_name](cal_true_traffics, cal_pred_traffics, alpha, test_pred_traffics[trial_idx], A, b, norm=2)
                values.append(value_trial)

                trial_runs[trial_idx] = value_trial
                trial_df = pd.DataFrame(trial_runs, index=[0])
                trial_df.to_csv(os.path.join(result_dir, f"{method_name}.csv"))

            if method_name not in method_values:
                method_values[method_name] = []
                method_std[method_name] = []

            method_values[method_name].append(np.mean(values))
            method_std[method_name].append(np.std(values))

    values_df = pd.DataFrame(method_values)
    std_df = pd.DataFrame(method_std)

    values_df.to_csv(os.path.join(result_dir, "values.csv"))
    std_df.to_csv(os.path.join(result_dir, "std.csv"))