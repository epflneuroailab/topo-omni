"""

this file contains infrastructure for position embeddings
 - partially built on top of spacetorch (github.com/neuroailab/TDANN)
 - infra for loading / saving positions
 - scripts for spatial loss calculation
 - scripts for pre-optimization

"""

import numpy as np
import pickle as pkl
from dataclasses import dataclass

from tqdm import tqdm

from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

### CLASS INFRA - LayerPositions + NetworkPositions ###

@dataclass
class LayerPositions:

    name: str # layer name

    # coordinates is an N x 2 matrix with the x-coordinates of each unit in the first
    # column and the y-coordinates in the second column
    coordinates: torch.Tensor

    # neighborhood_indices is a M x N binary matrix, where there are M neighborhoods
    neighborhood_indices: torch.Tensor

    # number of neighborhoods to compute / average loss over
    neighborhoods_per_batch: int

    def save(self, save_dir: Path):

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok = True, parents = True)
        path = save_dir / f'{self.name}.pkl'

        with path.open('wb') as f:
            pkl.dump(self, f)

    # load positions from disk
    @classmethod
    def load(cls, path: Path) -> 'LayerPositions':

        path = Path(path)

        assert path.exists(), path
        assert path.suffix == '.pkl', 'invalid file, needs to be pickle'

        with path.open('rb') as f:
            return pkl.load(f)

    # helper to put things on gpu
    def to(self, device):
        self.coordinates = self.coordinates.to(device)
        self.neighborhood_indices = self.neighborhood_indices.to(device)

        return self


@dataclass
class NetworkPositions:

    # dictionary of all layer positions
    layer_positions: Dict[str, LayerPositions]
    version: int

    # load ALL positions from disk
    @classmethod
    def load_from_dir(cls, load_dir: Path):

        load_dir = Path(load_dir)

        assert load_dir.is_dir(), load_dir
        layer_files = list(load_dir.glob('*.pkl'))

        d = {}

        for layer_file in layer_files:

            layer_name = layer_file.stem
            d[layer_name] = LayerPositions.load(layer_file)

        version_path = load_dir / 'version.txt'
        version = 1.0

        if version_path.is_file():
            with version_path.open('r') as f:
                version = int(f.readline())

        return cls(version = version, layer_positions = d)

### SPATIAL LOSS ###

# norm implementation for lp-norm
# for some reason this makes things 20x faster than torch.norm????
def p_norm(positions, p='inf'):

    diff = positions[:, None, :] - positions[None, :, :]
    
    if p == 1:
        return torch.sum(torch.abs(diff), dim=-1)
    if p == 2:
        return torch.sqrt(torch.sum(diff ** 2, dim=-1))
    if p == 'inf':
        return torch.max(torch.abs(diff), dim=-1)[0]

    raise ValueError(f'norm type {p} not supported')

def get_neighborhood(center, positions, radius = 5, p = 'inf'):
    distances = p_norm(positions, p)
    return distances[center] < radius

def local_spatial_loss(activations, positions):
    """
    activations: (B, K)  # B = batch/time samples, K = units/points in neighborhood
    positions:   (K, 2)
    """
    K = activations.shape[1]

    activations = activations.float()
    positions = positions.float()

    device = activations.device

    # Pairwise distance-based similarity D (K, K)
    dist = torch.cdist(positions, positions, p=2)          # Euclidean
    D = 1.0 / (1.0 + dist)

    # Correlation among units across the batch dimension (K, K)
    # activations = activations + 1e-6 * torch.randn_like(activations)
    r = torch.corrcoef(activations.T)

    # Lower-triangular indices (exclude diagonal)
    idx = torch.tril_indices(K, K, offset=-1, device=device)
    pairs_r = r[idx[0], idx[1]]
    pairs_D = D[idx[0], idx[1]]

    return 0.5 * (1.0 - torch.corrcoef(torch.stack([pairs_r, pairs_D], dim=0))[0, 1])

def spatial_loss_fn(activations, positions, accum='mean'):
    device = activations.device

    neigh_idx_all = positions.neighborhood_indices.to(device)  # (M, K) long
    coords = positions.coordinates.to(device)                  # (M, 2)

    num_units = neigh_idx_all.shape[0] # total number of units
    sampled = torch.randperm(num_units, device=device)[:positions.neighborhoods_per_batch]

    losses = []
    for i in sampled:
        idx = neigh_idx_all[i].squeeze(-1)     # (K,)
        idx = idx[idx >= 0]
        loss = local_spatial_loss(activations[:, idx], coords[idx])
        if not torch.isnan(loss):
            losses.append(loss)

    losses = torch.stack(losses) if losses else (activations.sum() * 0.0)
    if accum == 'mean':
        return losses.mean()
    elif accum == 'maximum':
        return losses.max()
    else:
        raise ValueError("invalid accumulation function")

# binary mask of points within 'radius' of center
def get_neighborhood(center, distances, radius = 5):
    return distances[center] < radius

# randomly select a center point for a neighborhood (avoiding the edges of the layer)
def get_center(positions, radius, rng=None):

    N = int(np.sqrt(positions.shape[0]))

    mask = (positions[:, 0] >= radius - 1) & (positions[:, 0] <= N - radius) & \
           (positions[:, 1] >= radius - 1) & (positions[:, 1] <= N - radius)

    indices = torch.nonzero(torch.Tensor(mask)).flatten()

    return indices[torch.randint(0, len(indices), (1,), generator = rng).item()]

def neighborhoods_chebyshev_grid(H, W, radius, device=None):
    k = 2 * radius - 1
    pad = radius - 1
    device = device or "cpu"

    idx = torch.arange(H * W, device=device).view(1, 1, H, W).long()

    # pad with -1 so "outside" is clearly invalid
    idx_pad = F.pad(idx, (pad, pad, pad, pad), mode="constant", value=-1)

    patches = F.unfold(idx_pad.float(), kernel_size=(k, k), padding=0, stride=1).long()
    neighborhoods = patches.T  # (H*W, k*k) because we padded and used padding=0

    return neighborhoods  # contains -1 for out-of-bounds

def p_norm_numpy(positions, p = 'inf'):

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    if p == 1:
        return np.sum(np.abs(diff), axis = -1)
    if p == 2:
        return np.sqrt(np.sum(diff ** 2, axis = -1))
    if p == 'inf':
        return np.max(np.abs(diff), axis = -1)

    raise ValueError(f'norm type {p} not supported')

def linf_dist_blocked(positions, block=4000):
    N = positions.shape[0]
    D = np.empty((N, N), dtype=positions.dtype)

    for i in tqdm(range(0, N, block)):
        i2 = min(i + block, N)
        positions_i = positions[i:i2]

        for j in range(0, N, block):
            j2 = min(j + block, N)
            positions_j = positions[j:j2]

            diff = positions_i[:, None, :] - positions_j[None, :, :]
            D[i:i2, j:j2] = np.max(np.abs(diff), axis=-1)

    return D

def local_spatial_loss_numpy(activations, positions):

    num_units = activations.shape[1]
    idx = np.tril_indices(num_units, k = -1)

    D = 1 / (1 + p_norm_numpy(positions, 2))
    r = np.corrcoef(activations.T)

    return 0.5 * (1 - np.corrcoef(r[idx], D[idx])[0][1])

def spatial_loss_numpy(activations, positions):

    cur_loss = []
    neighborhoods = np.random.choice(len(positions.neighborhood_indices), positions.neighborhoods_per_batch, replace = False)

    for i in neighborhoods:
        mask = positions.neighborhood_indices[i].astype(bool)
        cur_loss.append(local_spatial_loss_numpy(activations[:, mask], positions.coordinates[mask]))

    return np.stack(cur_loss).mean()