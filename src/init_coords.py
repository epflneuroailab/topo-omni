from typing import List

from dataclasses import dataclass

import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from itertools import product

from models.spatial_utils import LayerPositions, get_neighborhood, p_norm, p_norm_numpy, linf_dist_blocked, neighborhoods_chebyshev_grid


def permute_coordinates(coordinates, rng):
    # Define the three parts: (row_start, row_end, col_start, col_end, rows_per_layer)
    # parts = [
    #     (0, 144, 0, 512, 3),      # thinker
    #     (144, 304, 0, 256, 5),     # vision encoder
    #     (144, 304, 256, 512, 5),   # audio encoder
    # ]

    parts = [
        (0, 160, 0, 256, 5),     # vision encoder
        (0, 160, 256, 512, 5),   # audio encoder
        (160, 304, 0, 512, 3),   # thinker
    ]

    for row_start, row_end, col_start, col_end, rows_per_layer in parts:
        num_layers = (row_end - row_start) // rows_per_layer
        for layer_idx in range(num_layers):
            lr_start = row_start + layer_idx * rows_per_layer
            lr_end = lr_start + rows_per_layer

            # Collect flat indices of coordinates belonging to this layer
            indices = []
            for r in range(lr_start, lr_end):
                for c in range(col_start, col_end):
                    indices.append(r * N_col + c)

            indices = torch.tensor(indices)

            # Random permutation within this layer
            perm = torch.randperm(len(indices), generator=rng)

            # Apply the permutation
            original_values = coordinates[indices].clone()
            coordinates[indices] = original_values[perm]

    return coordinates


@dataclass
class PosConfig:
    model_name: str
    n_col: int
    n_row: int
    neighborhoods_per_batch: int
    radius: int
    p: str = 'inf'

    def __post_init__(self):
        self.num_units = self.n_col * self.n_row

if __name__ == "__main__":

    ### HYPERPARAMS ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SEED = 42

    rng = torch.Generator()
    rng.manual_seed(SEED)

    with open('configs/init_coords.yml', 'r') as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    radius = config["radius"]
    neighborhoods_per_batch = config["neighborhoods_per_batch"]
    n_col = config["n_col"]
    n_row = config["n_row"]

    pos_config = PosConfig(
        model_name=model_name,
        n_col=n_col,
        n_row=n_row,
        neighborhoods_per_batch=neighborhoods_per_batch,
        radius=radius,
        p=config.get("p", "inf"),
    )

    position_dir = f'neighborhoods/model={model_name}_radius={radius}_neighborhoods={neighborhoods_per_batch}_coords={SEED}'
    os.makedirs(position_dir, exist_ok=True)

    N_col = pos_config.n_col
    N_row = pos_config.n_row

    num_neighborhoods = (pos_config.n_row - 2 * pos_config.radius) * (pos_config.n_col - 2 * pos_config.radius)
    
    print(f"Multimodal Cortical Sheet: {pos_config.n_col} x {pos_config.n_row} = {pos_config.num_units} units")

    coordinates = torch.Tensor([(i,j) for i in range(N_row) for j in range(N_col)]).to(device)
    coordinates = permute_coordinates(coordinates, rng)

    name = f"unified_sheet"

    pos = LayerPositions(
        name = name,
        coordinates = coordinates,
        neighborhood_indices = None,
        neighborhoods_per_batch = neighborhoods_per_batch
    )

    mask = (
        (coordinates[:, 0] >= pos_config.radius) &
        (coordinates[:, 0] <= N_row - pos_config.radius - 1) &
        (coordinates[:, 1] >= pos_config.radius) &
        (coordinates[:, 1] <= N_col - pos_config.radius - 1)
    )

    indices = torch.nonzero(mask).flatten()

    num_neighborhoods = (pos_config.n_row - 2 * pos_config.radius) * (pos_config.n_col - 2 * pos_config.radius)
    num_units = pos_config.num_units
    radius_sq = pos_config.radius ** 2
    neighbors_per_center = (2 * pos_config.radius - 1) ** 2
    chunk_size = 1024

    neighborhood_indices = torch.full((num_units, neighbors_per_center), -1, dtype=torch.long, device=device)

    for start in tqdm(range(0, num_units, chunk_size)):
        end = min(start + chunk_size, num_units)
        chunk_len = end - start
        
        sq_dists = torch.cdist(coordinates[start:end], coordinates).pow(2)
        is_neighbor = sq_dists < radius_sq

        counts = is_neighbor.sum(dim=1)
        max_n = min(counts.max().item(), neighbors_per_center)

        sorted_vals, sorted_idx = is_neighbor.long().sort(dim=1, descending=True)
        neighborhood_indices[start:end, :max_n] = sorted_idx[:, :max_n]

        arange = torch.arange(neighbors_per_center, device=device).unsqueeze(0)
        valid = arange < counts.unsqueeze(1)
        neighborhood_indices[start:end][~valid] = -1

    pos.neighborhood_indices = neighborhood_indices.cpu()

    np.save(f"{position_dir}/coords.npy", pos.coordinates.cpu().numpy())
    np.save(f"{position_dir}/neighborhood_indices.npy", pos.neighborhood_indices.cpu().numpy())

    # pos.neighborhood_indices = neighborhoods_chebyshev_grid(N_row, N_col, pos_config.radius, device=device)