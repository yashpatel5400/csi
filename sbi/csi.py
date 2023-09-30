"""
To launch all the tasks, create tmux sessions and run across tasks, i.e.

python csi.py --task two_moons
python csi.py --task gaussian_mixture
"""

import time
import torch
import sbibm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import seaborn_image as isns
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

proj_dim = 4

class CSI:
    def __init__(self, prior, simulator, encoder, N, k, mins=None, maxs=None, desired_coverage = 0.95):
        self.prior     = prior
        self.simulator = simulator
        self.encoder   = encoder
        self.N         = N
        self.k         = k
        self.conformal_quantile = self.get_conformal_quantile(self.prior, self.simulator, self.encoder, desired_coverage=desired_coverage)
        self.trajectories = None
        self.test_samples = None

        self.mins = mins
        self.maxs = maxs
        if self.mins is None:
            self.mins = self.prior.support.base_constraint.lower_bound.cpu().numpy()
        if self.maxs is None:
            self.maxs = self.prior.support.base_constraint.upper_bound.cpu().numpy()
        self.mins = self.mins[:proj_dim]
        self.maxs = self.maxs[:proj_dim]

    def get_conformal_quantile(self, prior, simulator, encoder, desired_coverage):
        sims = 10_000
        calibration_theta = prior.sample((sims,))
        calibration_x = simulator(calibration_theta)
        calibration_theta = calibration_theta[:,:proj_dim]

        # cal_scores = 1 / encoder.log_prob(calibration_theta.to(device), calibration_x.to(device)).detach().cpu().exp().numpy()

        theta_cal_hat = encoder.sample(self.k, calibration_x).detach().cpu().numpy()
        theta_cal_tiled = np.transpose(np.tile(calibration_theta.detach().cpu().numpy(), (self.k, 1, 1)), (1, 0, 2))
        theta_cal_diff = theta_cal_hat - theta_cal_tiled
        theta_cal_norms = np.linalg.norm(theta_cal_diff, axis=-1)
        theta_cal_scores = np.min(theta_cal_norms, axis=-1)

        return np.quantile(theta_cal_scores, q = desired_coverage)
    
    def gen_test_samples(self, text_x):
        # not really a fan of this, but this caches the generated samples, since we wish to have a fixed region
        self.test_samples = self.encoder.sample(self.k, text_x).detach().cpu().numpy()[0]

    def _get_grid(self, K = 200):
        # K -> discretization of the grid (assumed same for each dimension)
        d = len(self.mins) # dimensionality of theta
        ranges = [np.arange(self.mins[i], self.maxs[i], (self.maxs[i] - self.mins[i]) / K) for i in range(d)]
        return np.array(np.meshgrid(*ranges)).T.astype(np.float32)

    def _get_conformal_region(self, test_x, thetas):
        thetas_flat = thetas.reshape(-1, thetas.shape[-1])
        theta_tiled = np.transpose(np.tile(thetas_flat, (self.k, 1, 1)), (1, 0, 2))
        theta_diff = self.test_samples - theta_tiled
        theta_norm = np.linalg.norm(theta_diff, axis=-1)
        theta_score = np.min(theta_norm, axis=-1)

        # test_x_tiled = np.tile(test_x, (thetas_flat.shape[0], 1)).astype(np.float32)
        # probs = self.encoder.log_prob(thetas_flat, test_x_tiled).detach().cpu().exp().numpy()
        
        flat_region = (theta_score < self.conformal_quantile).astype(int)
        return flat_region.reshape(thetas.shape[:-1])

    def _get_connected_components(self, region):
        active_region = np.where(region == 1)
        active_locs = set(zip(*active_region))

        components = []
        while len(active_locs) > 0:
            component = set()
            to_branch = {active_locs.pop()}
            while len(to_branch) > 0:
                current_loc = to_branch.pop()
                component.add(current_loc)
                current_loc_arr = np.array(current_loc) # necessary to change component values
                for dim in range(len(current_loc_arr)):
                    for displacement in [-1, 1]:
                        displacement_loc = np.zeros(len(current_loc_arr)).astype(int)
                        displacement_loc[dim] = displacement
                        candidate_loc = tuple(current_loc_arr + displacement_loc)
                        if candidate_loc in active_locs:
                            active_locs.remove(candidate_loc)
                            to_branch.add(candidate_loc)
            components.append(np.array(list(component)))
        return components

    def _get_rps_cc(self, regions_samples):
        total_covered = np.sum([len(region_samples) for region_samples in regions_samples])
        total_rps = 0
        rps = []

        for region_idx, region_samples in enumerate(regions_samples):
            component_prop = len(region_samples) / total_covered
            if region_idx == len(regions_samples) - 1:
                n = self.N - total_rps
            else:
                n = int(np.round(component_prop * self.N))
                total_rps += n

            # TODO: should we ensure each connected component is > 1 in the "exact answer"? feels arbitrary but maybe desireable?
            if n > 0:
                kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(region_samples)
                rps.append(kmeans.cluster_centers_)
        return np.vstack(rps)

    def get_exact_rps(self, test_x):
        K = 60
        theta_grid = self._get_grid(K=K)
        region = self._get_conformal_region(test_x, theta_grid)
        self.explicit_region = theta_grid[np.where(region == 1)] # cache explicit region

        connected_components = self._get_connected_components(region)
        region_samples = [theta_grid[tuple(connected_component.T)] for connected_component in connected_components]
        return self._get_rps_cc(region_samples)

    def _get_diffused_trajs(self, test_x, T):
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

    def get_approx_rps_diffuse(self, test_x, T=5_000, cache_trajs=False):
        """
        NOTE: this form of uniform sampling is now deprecated in favor of Muller's sampling

        T : time steps of repulsive simulation
        cache_trajs: HACK -- this arg doesn't really make any sense in real use cases, but it caches the trajectories
            between calls of this function -- it should ONLY be used for repeated calls to this for visualization
        """
        if cache_trajs and self.trajectories is not None:
            trajectories = self.trajectories[:T]
        else:
            trajectories = self._get_diffused_trajs(test_x, T)
            self.trajectories = trajectories

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
    
    def _mullers_sample_from_ball(self, center, r, N, d):
        u = np.random.normal(0, 1, (N, d))
        norm = np.linalg.norm(u, axis=1)
        radius = np.random.uniform(0, 1, N) ** (1/d)
        
        u = r * radius.reshape(-1, 1) * u / norm.reshape(-1, 1)
        u = u + center
        return u

    def get_approx_rps(self, N):
        edges = []
        while len(edges) == 0:
            d = self.test_samples.shape[-1]
            
            samples = np.zeros((self.k, N, d))
            voronoi_assignments = np.zeros((self.k, N))

            # fraction of samples in each ball in the associated voronoi cell
            samples = np.apply_along_axis(self._mullers_sample_from_ball, axis=1, arr=self.test_samples, r=self.conformal_quantile, N=N, d=d)
            samples = samples.reshape(self.k, N, 1, -1)
            dist = np.linalg.norm(samples - self.test_samples, axis=-1)
            voronoi_assignments = np.argmin(dist, axis=-1).flatten()
            sampled_assignments = np.array([[k] * N for k in range(self.k)]).flatten()
            
            flat_samples = samples.reshape(-1, *samples.shape[-2:])
            samples = np.vstack(flat_samples[np.where(voronoi_assignments == sampled_assignments),0,:])
            
            # voronoi_regions_samples = [flat_samples[voronoi_assignments == voronoi_idx] for voronoi_idx in range(self.k)]
            # min_voronoi_samples = np.min([len(voronoi_region_samples) for voronoi_region_samples in voronoi_regions_samples])
            # samples = np.vstack([voronoi_region_samples[:min_voronoi_samples,0,:] for voronoi_region_samples in voronoi_regions_samples])
            
            # create graph
            kdt = spatial.KDTree(samples)
            edges = kdt.query_pairs(self.conformal_quantile)
        G = nx.from_edgelist(edges)

        connected_components = list(nx.connected_components(G))        
        region_samples = [samples[np.array(list(connected_component))] 
                          for connected_component in connected_components]
        return self._get_rps_cc(region_samples)
    
    def viz_rps(self, test_x, exact_rps, approx_rps, fn):
        K = 100
        theta_grid = self._get_grid(K=K)
        region = self._get_conformal_region(test_x, theta_grid)
        region = region.reshape((K, K))
        
        plt.imshow(region, extent=[self.mins[0], self.maxs[0], self.mins[1], self.maxs[1]], origin="lower")
        sns.scatterplot(x=approx_rps[:,1], y=approx_rps[:,0], s=10, palette="deep", label=r"$\widehat{\Xi}$")
        sns.scatterplot(x=exact_rps[:,1], y=exact_rps[:,0], s=10, palette="deep", label=r"$\Xi$")

        plt.xticks([])
        plt.yticks([])
        plt.xlim(self.mins[0], self.maxs[0])
        plt.ylim(self.mins[1], self.maxs[1])

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
    
    def get_rps_obj(self, rps):
        theta_tiled = np.transpose(np.tile(self.explicit_region, (rps.shape[0], 1, 1)), (1, 0, 2))
        theta_diff = rps - theta_tiled
        theta_norm = np.linalg.norm(theta_diff, axis=-1)
        theta_score = np.min(theta_norm, axis=-1)
        return np.mean(theta_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    args = parser.parse_args()
    task_name = args.task

    device = "cpu"
    task = sbibm.get_task(task_name)
    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    cached_fn = os.path.join("projected_results", f"{task_name}.nf")
    with open(cached_fn, "rb") as f:
        encoder = pickle.load(f)
    encoder.to(device)

    # sample_idx = 0
    # while sample_idx < 10:
    for sample_idx in range(10):
        test_sim = 1
        test_theta = prior.sample((test_sim,))
        test_x = simulator(test_theta)
        test_theta = test_theta[:,:proj_dim]

        with open(os.path.join("minmax", f"{task_name}.pkl"), "rb") as f:
            mins, maxs = pickle.load(f)

        Ns = np.arange(5, 501, 5)
        csi = CSI(prior=prior, simulator=simulator, encoder=encoder, N=5, k=10, mins=mins, maxs=maxs, desired_coverage=0.95)
        os.makedirs("results", exist_ok=True)

        csi.gen_test_samples(test_x)

        print("Computing exact RPs...")
        exact_rps  = csi.get_exact_rps(test_x)
        exact_obj = csi.get_rps_obj(exact_rps)
                    
        # if sample_idx == 0:
        #     print("Performing visualization...")
        #     approx_rps = csi.get_approx_rps(N=20)
        #     csi.viz_rps(test_x, exact_rps, approx_rps, os.path.join("results", f"{task_name}_rps.png"))

        print("Computing approximate RPs...")
        optimality_gaps = []
        for N in Ns:
            print(f"Computing {N}...")
            approx_rps = csi.get_approx_rps(N=N)
            optimality_gaps.append(csi.get_rps_obj(approx_rps) - exact_obj)

        with open(os.path.join("results", "dists", f"{task_name}_{sample_idx}.pkl"), "wb") as f:
            pickle.dump((Ns, optimality_gaps), f)
        # sample_idx += 1