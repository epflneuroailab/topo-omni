from typing import List

from dataclasses import dataclass

import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
from itertools import product

from src.models.spatial_utils import (
    DEFAULT_SHEET_LAYOUT,
    LayerPositions,
    SheetLayout,
    build_sheet_coordinates,
    get_neighborhood,
    p_norm,
    p_norm_numpy,
    linf_dist_blocked,
    neighborhoods_chebyshev_grid,
)



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

    # Optional config path, so a second layout does not need this file edited in place.
    cfg_path = 'configs/init/coords.yml'
    if "--config" in sys.argv:
        cfg_path = sys.argv[sys.argv.index("--config") + 1]
    print(f"config: {cfg_path}")
    with open(cfg_path, 'r') as f:
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

    # Take the layout from $TOPO_SHEET_LAYOUT rather than rebuilding a bare
    # SheetLayout(n_row, n_col). The coordinates encode the BAND STRUCTURE -- how many
    # rows each encoder slot and thinker layer owns, and the shuffle applied within each
    # of those blocks -- not just the sheet dimensions. A bare SheetLayout would silently
    # use the 309x512 defaults (encoder_rows=165, 5 rows per slot) and produce
    # coordinates whose blocks do not line up with the layer the model actually writes
    # there. For the legacy 309x512 strip this is identical to what it did before.
    layout = DEFAULT_SHEET_LAYOUT
    if (layout.n_row, layout.n_col) != (N_row, N_col):
        raise SystemExit(
            f"config says {N_row}x{N_col} but TOPO_SHEET_LAYOUT="
            f"{os.environ.get('TOPO_SHEET_LAYOUT', 'strip')!r} is "
            f"{layout.n_row}x{layout.n_col}. Fix whichever is wrong; a mismatch here "
            f"writes coordinates the model cannot use."
        )
    # The original strip pins 3 thinker rows per shuffle block purely to stay
    # bit-identical with the coordinates this script has always produced. Any other
    # layout should shuffle within the block a thinker layer actually occupies.
    thinker_block = 3 if (N_row, N_col) == (309, 512) else layout.rows_per_thinker_layer
    print(f"layout: mode={layout.mode} {layout.n_row}x{layout.n_col} "
          f"cortical_dim={layout.cortical_dim} units/layer={layout.units_per_layer} "
          f"enc_rows/layer={layout.rows_per_encoder_layer} thinker_block={thinker_block}")
    coordinates = build_sheet_coordinates(
        layout, seed=SEED, thinker_rows_per_block=thinker_block).to(device)

    np.save(f"{position_dir}/coords.npy", coordinates.cpu().numpy())

    if "--coords-only" in sys.argv:
        # the dynamic-neighborhood training path resolves neighborhoods on the fly
        # and only needs coords.npy, so skip the expensive table build
        print(f"Wrote coordinates only to {position_dir}")
        sys.exit(0)

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

    np.save(f"{position_dir}/neighborhood_indices.npy", pos.neighborhood_indices.cpu().numpy())

    # pos.neighborhood_indices = neighborhoods_chebyshev_grid(N_row, N_col, pos_config.radius, device=device)