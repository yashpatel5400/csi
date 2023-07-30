import time
import torch
import sbibm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from scipy.ndimage.measurements import label
from sklearn.cluster import KMeans
from scipy import spatial
import itertools
import argparse

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

class CSI:
    def __init__(self, prior, simulator, encoder, N, desired_coverage = 0.95):
        self.prior     = prior
        self.simulator = simulator
        self.encoder   = encoder
        self.N         = N
        self.conformal_quantile = self.get_conformal_quantile(self.prior, self.simulator, self.encoder, desired_coverage=desired_coverage)

    def get_conformal_quantile(self, prior, simulator, encoder, desired_coverage):
        sims = 10_000 # same number for both test and calibration
        calibration_theta = prior.sample((sims,))
        calibration_x = simulator(calibration_theta)
        cal_scores = 1 / encoder.log_prob(calibration_theta.to(device), calibration_x.to(device)).detach().cpu().exp().numpy()
        return np.quantile(cal_scores, q = desired_coverage)

    def _get_grid(self, K = 200):
        # K -> discretization of the grid (assumed same for each dimension)
        mins = self.prior.support.base_constraint.lower_bound.cpu().numpy()
        maxs = self.prior.support.base_constraint.upper_bound.cpu().numpy()
        d = len(mins) # dimensionality of theta
        ranges = [np.arange(mins[i], maxs[i], (maxs[i] - mins[i]) / K) for i in range(d)]
        return np.array(np.meshgrid(*ranges)).T.reshape(-1, d).astype(np.float32)

    def _get_conformal_region(self, test_x, thetas):
        test_x_tiled = np.tile(test_x, (thetas.shape[0], 1)).astype(np.float32)
        probs = self.encoder.log_prob(thetas, test_x_tiled).detach().cpu().exp().numpy()
        return ((1 / probs) < self.conformal_quantile).astype(int)

    def get_exact_rps(self, test_x):
        K = 200
        theta_grid = self._get_grid(K=K)
        region = self._get_conformal_region(test_x, theta_grid)
        region = region.reshape((K, K))
        
        structure = np.ones((3, 3), dtype=int)  # this defines the connection filter
        labeled, ncomponents = label(region, structure)

        total_covered = np.sum(region)
        total_rps = 0
        exact_rps = []

        for component in range(1, ncomponents + 1):
            component_prop = np.sum(labeled == component) / total_covered
            if component == ncomponents:
                n = self.N - total_rps
            else:
                n = int(np.round(component_prop * self.N))
                total_rps += n

            # TODO: should we ensure each connected component is > 1 in the "exact answer"? feels arbitrary but maybe desireable?
            if n > 0:
                grid_region = theta_grid[(labeled == component).flatten()]
                kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(grid_region)
                exact_rps.append(kmeans.cluster_centers_)
        return np.vstack(exact_rps)

    def _get_diffused_trajs(self, test_x):
        T = 5_000 # time steps of repulsive simulation

        y_hat = self.encoder.sample(self.N, test_x)[0].detach().cpu().numpy()
        test_x_tiled = np.tile(test_x, (y_hat.shape[0], 1)).astype(np.float32)
        trajectories = []
        eta = 0.01

        for _ in range(T):
            proposed_y_hat = y_hat.copy() + eta * np.random.randn(y_hat.shape[0], y_hat.shape[1]).astype(np.float32)
            proposed_probs = self.encoder.log_prob(proposed_y_hat, test_x_tiled).detach().cpu().exp().numpy()

            in_region = (1 / proposed_probs) < self.conformal_quantile
            y_hat[in_region] = proposed_y_hat[in_region]
            trajectories.append(y_hat.copy())
        return np.array(trajectories)

    def get_approx_rps(self, test_x):
        trajectories = self._get_diffused_trajs(test_x)

        remaining_traj_idxs = set(range(self.N))
        final_trajs = []
        ns = []
        dist_thresh = 0.01
        while len(remaining_traj_idxs) > 0:
            root_traj_idx = remaining_traj_idxs.pop()
            connected_trajs = [trajectories[:,root_traj_idx,:]]
            connected_traj_idxs = set()
            
            tree = spatial.KDTree(trajectories[:,root_traj_idx,:])
            for traj_idx in remaining_traj_idxs:
                min_dists, _ = tree.query(trajectories[:,traj_idx,:])
                closest_encounter = min(min_dists)
                if closest_encounter < dist_thresh:
                    connected_trajs.append(trajectories[:,traj_idx,:])
                    connected_traj_idxs.add(traj_idx)
            
            for traj_idx in connected_traj_idxs:
                if traj_idx in remaining_traj_idxs:
                    remaining_traj_idxs.remove(traj_idx)
            ns.append(len(connected_trajs))
            final_trajs.append(np.vstack(connected_trajs))

        rps = []
        for traj_idx, final_traj in enumerate(final_trajs):
            kmeans = KMeans(n_clusters=ns[traj_idx], random_state=0, n_init="auto").fit(final_traj)
            rps.append(kmeans.cluster_centers_)
        return np.vstack(rps)
    
    def viz_rps(self, test_x, approx_rps, exact_rps, fn):
        K = 200
        theta_grid = self._get_grid(K=K)
        region = self._get_conformal_region(test_x, theta_grid)
        region = region.reshape((K, K))
        
        mins = self.prior.support.base_constraint.lower_bound.cpu().numpy()
        maxs = self.prior.support.base_constraint.upper_bound.cpu().numpy()
        
        plt.imshow(region, extent=[mins[0], maxs[0], mins[1], maxs[1]], origin="lower")
        plt.scatter(approx_rps[:,1], approx_rps[:,0], s=10, color="red", label="Estimate")
        plt.scatter(exact_rps[:,1], exact_rps[:,0], s=10, color="green", label="Exact")

        plt.xticks([])
        plt.yticks([])
        plt.xlim(mins[0], maxs[0])
        plt.ylim(mins[1], maxs[1])
        plt.legend()

        plt.savefig(fn)
        plt.clf()

    def _list_assoc(self, L1, L2, reverse=False):
        tree = spatial.KDTree(L2)
        assoc = []
        for I1, point in enumerate(L1):
            _, I2 = tree.query(point,k=1)
            if reverse:
                assoc.append((I2, I1))
            else:
                assoc.append((I1, I2))
        return assoc

    def _get_opt_correspondence(self, exact_rps, approx_rps):
        # some complication to ensure 1-1 correspondence
        exact_to_approx_assoc_A = self._list_assoc(exact_rps, approx_rps, reverse=False)
        exact_to_approx_assoc_B = self._list_assoc(approx_rps, exact_rps, reverse=True)
        exact_to_approx_assoc   = list(set.intersection(set(exact_to_approx_assoc_A), set(exact_to_approx_assoc_B)))
        
        # exhaustively enumerate and match remainder: N here should be small, so N! should be manageable
        unmatched_exact = list(range(len(exact_rps)))
        unmatched_approx = list(range(len(approx_rps)))
        [(unmatched_exact.remove(assoc[0]), unmatched_approx.remove(assoc[1])) for assoc in exact_to_approx_assoc]
        
        if len(unmatched_exact) > 0:
            approx_permutations = list(itertools.permutations(unmatched_approx))
            dists = []
            for approx_permutation in approx_permutations:
                correspondence_dist = exact_rps[unmatched_exact] - approx_rps[approx_permutation,:]
                dists.append(np.sum(correspondence_dist ** 2))
            unmatched_opt = np.argmin(dists)

            exact_to_approx_assoc += list(zip(unmatched_exact, approx_permutations[unmatched_opt]))
        exact_to_approx_assoc = np.array(sorted(exact_to_approx_assoc)).astype(np.int32)
        return exact_to_approx_assoc

    def get_dist(self, exact_rps, approx_rps):
        exact_to_approx_assoc = self._get_opt_correspondence(exact_rps, approx_rps)
        opt_correspondence_dist = exact_rps[exact_to_approx_assoc[:,0]] - approx_rps[exact_to_approx_assoc[:,1]]
        return np.sum(opt_correspondence_dist ** 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    args = parser.parse_args()
    task_name = args.task

    device = "cpu"
    task = sbibm.get_task(task_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    cached_fn = os.path.join("trained", f"{task_name}.nf")
    with open(cached_fn, "rb") as f:
        encoder = pickle.load(f)
    encoder.to(device)

    test_sim = 1
    test_theta = prior.sample((test_sim,))
    test_x = simulator(test_theta)

    csi = CSI(prior=prior, simulator=simulator, encoder=encoder, N=8, desired_coverage=0.95)

    print("Computing exact RPs...")
    exact_rps  = csi.get_exact_rps(test_x)

    print("Computing approximate RPs...")
    approx_rps = csi.get_approx_rps(test_x)

    print("Performing visualization...")
    csi.viz_rps(test_x, exact_rps, approx_rps, f"{task_name}_rps.png")

    print("Computing distances...")
    dist = csi.get_dist(exact_rps, approx_rps)
    print(dist)