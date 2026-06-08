# -------------------------------------------------
# Smoothing (HRF-like) with 3σ truncation
# -------------------------------------------------

from typing import Tuple

import numpy as np

from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def _hrf_grid_shape(gridx: np.ndarray, gridy: np.ndarray) -> Tuple[int, int]:
    xs = np.unique(gridx.reshape(-1))
    ys = np.unique(gridy.reshape(-1))
    return ys.size+1, xs.size+1  # (H, W)


class NeuronSmoothingConv:
    """
    Same intention as NeuronSmoothing, but implemented via 2D Gaussian convolution
    instead of an explicit dense Gaussian filter matrix.
    """

    def __init__(self, fwhm_mm: float, resolution_mm: float):
        self.fwhm_mm = fwhm_mm
        self.resolution_mm = resolution_mm

        # Same sigma as your old code (in mm)
        self.sigma_mm = fwhm_mm / np.sqrt(8.0 * np.log(2.0))
        # Convert from mm to pixels (grid steps)
        self.sigma_pix = self.sigma_mm / resolution_mm

        cache_dir = "/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/bkhmsi/topoomni/cache"
        self.gridx = np.load(f"{cache_dir}/gridx.npy")
        self.gridy = np.load(f"{cache_dir}/gridy.npy")
        self.height, self.width = _hrf_grid_shape(self.gridx, self.gridy)

    def __call__(self, positions=None, activations=None):
        """
        positions:  (N_neurons, 2) -> (x, y) in mm, on a regular grid
        activations: (N_samples, N_neurons)

        Returns:
          gridx, gridy: (N_grid, 1) each
          features_smoothed: (N_samples, N_grid)
        """

        def _get_grid_coord(x, y):
            xmin, xmax = np.floor(np.min(x)), np.ceil(np.max(x))
            ymin, ymax = np.floor(np.min(y)), np.ceil(np.max(y))
            xs = np.arange(xmin, xmax + 1, self.resolution_mm)
            ys = np.arange(ymin, ymax + 1, self.resolution_mm)
            return xs, ys

        tissue_x = positions[:, 0]
        tissue_y = positions[:, 1]

        xs, ys = _get_grid_coord(tissue_x, tissue_y)

        H, W = len(ys), len(xs)           
        N_samples, N_neurons = activations.shape
        assert N_neurons == len(positions)

        # Map each neuron to its grid index (row, col)
        xmin, ymin = xs[0], ys[0]
        col_idx = ((tissue_x - xmin) / self.resolution_mm).round().astype(int)
        row_idx = ((tissue_y - ymin) / self.resolution_mm).round().astype(int)

        # Build 2D activations on the grid for each sample
        acts_2d = np.zeros((N_samples, H, W), dtype=float)
        for n in range(N_neurons):
            r = row_idx[n]
            c = col_idx[n]
            acts_2d[:, r, c] += activations[:, n]

        # Apply Gaussian smoothing via convolution
        smoothed_2d = np.empty_like(acts_2d)
        for i in tqdm(range(N_samples)):
            smoothed_2d[i] = gaussian_filter(
                acts_2d[i],
                sigma=self.sigma_pix,
                truncate=3.0,     
                mode="nearest",   
                cval=0.0
            )

        features_smoothed = smoothed_2d.reshape(N_samples, H * W)

        return features_smoothed


class NeuronSmoothing:
    """
    Project per-neuron activations to a regular 2D grid using a Gaussian kernel,
    truncated at 3 * sigma (beyond that contributions are set to zero).
    """

    def __init__(self, fwhm_mm: float, resolution_mm: float):
        self.fwhm_mm = fwhm_mm
        self.resolution_mm = resolution_mm

    def __call__(self, positions=None, activations=None):
        """
        positions:  (N_neurons, 2)  -> (x, y) coordinates
        activations: (N_samples, N_neurons)

        Returns:
          gridx, gridy: (N_grid, 1) each
          features_smoothed: (N_samples, N_grid)
        """

        def _get_grid_coord(x, y):
            xmin, xmax = np.floor(np.min(x)), np.ceil(np.max(x))
            ymin, ymax = np.floor(np.min(y)), np.ceil(np.max(y))
            grids = np.array(np.meshgrid(
                np.arange(xmin, xmax + 1, self.resolution_mm),
                np.arange(ymin, ymax + 1, self.resolution_mm)
            ))
            gridx = grids[0].flatten().reshape(-1, 1)
            gridy = grids[1].flatten().reshape(-1, 1)
            return gridx, gridy

        tissue_x = positions[:, 0][:, np.newaxis] 
        tissue_y = positions[:, 1][:, np.newaxis] 
        gridx, gridy = _get_grid_coord(tissue_x, tissue_y) 

        sigma = self.fwhm_mm / np.sqrt(8.0 * np.log(2.0))

        d_square = (tissue_x - gridx.T) ** 2 + (tissue_y - gridy.T) ** 2

        gaussian_filter = np.zeros_like(d_square, dtype=float)
        mask = d_square <= (3.0 * sigma) ** 2
        gaussian_filter[mask] = (
            1.0 / (2.0 * np.pi * sigma ** 2)
            * np.exp(-d_square[mask] / (2.0 * sigma ** 2))
        )

        features_smoothed = np.dot(activations, gaussian_filter)  # (N_samples, N_grid)

        return gridx, gridy, features_smoothed