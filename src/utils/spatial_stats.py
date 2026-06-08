#!/usr/bin/env python3
"""
Spatial Statistics Utilities for fMRI Analysis
===============================================

Unified implementation of Moran's I spatial autocorrelation metrics.
Used by visualization and analysis scripts to ensure consistency.

This module implements two Moran's I metrics:

1. STANDARD MORAN'S I:
   - Computed on full unthresholded t-maps
   - Uses all vertices
   - Single global measure of spatial autocorrelation

2. ISLAND MORAN'S I:
   - Computed on FDR-corrected significant vertices only
   - Finds connected components (islands)
   - Computes Moran's I separately for each island (≥ min_size vertices)
   - Returns average Moran's I across islands

Reference Implementation:
- Adapted from user's island_morans_I reference code
- Uses brain surface mesh topology instead of lat2W grid
- Canonical implementation from script 27_compute_morans_i_both_methods.py

Dependencies:
- libpysal, esda (for Moran's I computation)
- scipy (sparse matrices, connected components)
- nibabel (surface mesh loading)
- nilearn (fsaverage surfaces)
- numpy

Author: Unified from scripts 17, 25, 27
Date: 2026-02-10
"""

import numpy as np
from tqdm import tqdm
from scipy import stats
from scipy.sparse.csgraph import connected_components

# Try to import libpysal and esda
try:
    import libpysal as lp
    from esda.moran import Moran
    from libpysal.weights import W as libpysal_W
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False
    print("WARNING: libpysal/esda not available. Moran's I computation will not work.")
    print("         Install with: pip install libpysal esda")

# Default parameters
DEFAULT_N_PERMUTATIONS = 999
DEFAULT_FDR_Q = 0.05
DEFAULT_MIN_ISLAND_SIZE = 8
DEFAULT_N_SUBJECTS = 83
DEFAULT_DF = DEFAULT_N_SUBJECTS - 1


def adjacency_to_libpysal_W(adj_matrix):
    """
    Convert scipy sparse adjacency matrix to libpysal W object.
    
    Parameters
    ----------
    adj_matrix : scipy.sparse.csr_matrix
        Adjacency matrix
    
    Returns
    -------
    libpysal.weights.W or None
        Libpysal W object, or None if conversion fails
    
    Notes
    -----
    This is a helper function used by compute_standard_morans_i.
    Creates unweighted spatial weights (all weights = 1.0).
    """
    if not LIBPYSAL_AVAILABLE:
        return None
    
    neighbors = {}
    weights = {}
    
    for i in range(adj_matrix.shape[0]):
        neighbor_indices = adj_matrix[i].nonzero()[1]
        if len(neighbor_indices) > 0:
            neighbors[i] = neighbor_indices.tolist()
            weights[i] = [1.0] * len(neighbor_indices)
    
    if len(neighbors) == 0:
        return None
    
    return libpysal_W(neighbors, weights)


def compute_standard_morans_i(t_map, n_permutations=DEFAULT_N_PERMUTATIONS):
    """
    Compute standard Moran's I on unthresholded t-map.
    
    This computes global spatial autocorrelation across all vertices.
    No thresholding is applied - uses the full t-statistic map.
    
    Parameters
    ----------
    t_map : np.ndarray
        T-statistic values for all vertices (1D array)
    n_permutations : int, default=999
        Number of permutations for significance test
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'I': float, Moran's I statistic
        - 'p_value': float, permutation-based p-value
        - 'n_vertices': int, number of vertices used
    
    Notes
    -----
    Uses libpysal's Moran class for computation.
    Returns NaN if computation fails or no valid neighbors exist.
    
    References
    ----------
    Moran, P. A. (1950). Notes on continuous stochastic phenomena.
    Biometrika, 37(1/2), 17-23.
    """


    t_map_mask = ~np.isnan(t_map)
    w = lp.weights.lat2W(t_map.shape[0], t_map.shape[1], rook=False)
    valid_indices = np.where(t_map_mask.flatten())[0]

    filtered_neighbors = {i: [j for j in w.neighbors[i] if j in valid_indices] for i in valid_indices}
    filtered_weights = {i: [w.weights[i][j_idx] for j_idx, j in enumerate(w.neighbors[i]) if j in valid_indices] for i in valid_indices}
    
    # Create a W object from the filtered neighbors and weights
    filtered_w = libpysal_W(filtered_neighbors, filtered_weights)

    # Create the adjacency matrix
    adj_matrix = filtered_w.sparse

    if not LIBPYSAL_AVAILABLE:
        return {'I': np.nan, 'p_value': np.nan, 'n_vertices': 0}
    
    # Convert adjacency to libpysal W
    w = adjacency_to_libpysal_W(adj_matrix)
    
    if w is None:
        return {'I': np.nan, 'p_value': np.nan, 'n_vertices': 0}
    
    # Compute Moran's I
    try:
        moran = Moran(t_map, w, permutations=n_permutations)
        return {
            'I': moran.I,
            'p_value': moran.p_sim,
            'n_vertices': len(t_map)
        }
    except Exception as e:
        print(f"    WARNING: Standard Moran's I failed: {e}")
        return {'I': np.nan, 'p_value': np.nan, 'n_vertices': len(t_map)}


def compute_fdr_threshold(t_map, p_map, q=DEFAULT_FDR_Q, df=DEFAULT_DF):
    """
    Calculate FDR threshold using Benjamini-Hochberg procedure (one-tailed).
    
    Parameters
    ----------
    t_map : np.ndarray
        T-statistic values (1D array)
    p_map : np.ndarray
        P-values (1D array, same length as t_map)
    q : float, default=0.05
        FDR threshold (false discovery rate)
    df : int, default=82
        Degrees of freedom (n_subjects - 1)
    
    Returns
    -------
    float
        T-statistic threshold for FDR correction.
        Returns np.inf if no vertices pass FDR threshold.
    
    Notes
    -----
    Only considers positive t-values (one-tailed test).
    Uses Benjamini-Hochberg procedure for FDR correction.
    
    References
    ----------
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
    a practical and powerful approach to multiple testing. Journal of the Royal
    Statistical Society: Series B (Methodological), 57(1), 289-300.
    """
    # Only consider positive t-values (one-tailed)
    positive_mask = t_map > 0
    
    if not np.any(positive_mask):
        return np.inf
    
    # One-tailed p-values for positive direction
    p_values_pos = p_map[positive_mask]
    
    # Sort p-values
    sorted_p = np.sort(p_values_pos)
    n_tests = len(sorted_p)
    
    # Benjamini-Hochberg procedure
    significant_mask = sorted_p <= (np.arange(1, n_tests + 1) / n_tests) * q
    
    if np.any(significant_mask):
        fdr_p_threshold = sorted_p[significant_mask][-1]
        # Convert p-threshold back to t-threshold (one-tailed)
        fdr_t_threshold = stats.t.ppf(1 - fdr_p_threshold, df=df)
        return fdr_t_threshold
    else:
        return np.inf


def compute_island_morans_i(t_map, p_map, fdr_q=DEFAULT_FDR_Q, 
                            min_size=DEFAULT_MIN_ISLAND_SIZE, 
                            n_permutations=DEFAULT_N_PERMUTATIONS,
                            df=DEFAULT_DF):
    """
    Compute Island's Moran's I on FDR-significant vertices.
    
    This implementation:
    1. Applies FDR threshold to identify significant vertices
    2. Finds connected components (islands) among significant vertices
    3. Computes Moran's I separately for each island ≥ min_size
    4. Returns average Moran's I across all islands
    
    This is adapted from the user's island_morans_I reference implementation,
    modified to work with brain surface mesh topology instead of lat2W grid.
    
    Parameters
    ----------
    t_map : np.ndarray
        T-statistic values (1D array)
    p_map : np.ndarray
        P-values (1D array)
    fdr_q : float, default=0.05
        FDR threshold
    min_size : int, default=8
        Minimum island size (vertices). Islands smaller than this are skipped.
    n_permutations : int, default=999
        Permutations for significance test
    df : int, default=82
        Degrees of freedom (n_subjects - 1) for FDR calculation
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'I': float, average Moran's I across ALL islands (unweighted mean)
        - 'p_value': float, average p-value across islands
        - 'n_islands': int, number of islands used
        - 'n_vertices': int, total number of significant vertices
        - 'island_details': list of dicts, per-island statistics
        - 'I_sig_only': float, average Moran's I across ONLY islands with p < 0.05 (unweighted)
        - 'n_sig_morans_i': int, count of islands with significant Moran's I (p < 0.05)
        - 'I_weighted_all': float, weighted average across ALL islands (weighted by island size)
        - 'I_weighted_sig': float, weighted average across ONLY sig islands (weighted by island size)
    
    Notes
    -----
    Reference implementation from user's island_morans_I function.
    Adapted for brain mesh topology (surface vertices) instead of lat2W grid.
    
    The algorithm:
    - Filter adjacency graph to only significant vertices
    - Use scipy.sparse.csgraph.connected_components to find islands
    - For each island ≥ min_size: compute Moran's I with permutation test
    - Average Moran's I and p-values across islands
    
    Returns NaN if no islands meet the size threshold.
    """
    if not LIBPYSAL_AVAILABLE:
        return {
            'I': np.nan,
            'p_value': np.nan,
            'n_islands': 0,
            'n_vertices': 0,
            'island_details': [],
            'I_sig_only': np.nan,
            'n_sig_morans_i': 0,
            'I_weighted_all': np.nan,
            'I_weighted_sig': np.nan
        }
    

    t_map_flatten = t_map.flatten()

    # Get FDR threshold
    fdr_t_threshold = compute_fdr_threshold(t_map_flatten, p_map.flatten(), q=fdr_q, df=df)
    
    if np.isinf(fdr_t_threshold):
        return {
            'I': np.nan,
            'p_value': np.nan,
            'n_islands': 0,
            'n_vertices': 0,
            'island_details': [],
            'I_sig_only': np.nan,
            'n_sig_morans_i': 0,
            'I_weighted_all': np.nan,
            'I_weighted_sig': np.nan
        }
    
    # Mask for FDR-significant vertices (positive only)
    sig_mask = (t_map_flatten >= fdr_t_threshold)
    t_threshold = np.percentile(t_map_flatten, 99)
    t_map_mask = sig_mask & (t_map_flatten >= t_threshold)
    valid_indices = np.where(t_map_mask)[0]
    
    if len(valid_indices) == 0:
        return {
            'I': np.nan,
            'p_value': np.nan,
            'n_islands': 0,
            'n_vertices': 0,
            'island_details': [],
            'I_sig_only': np.nan,
            'n_sig_morans_i': 0,
            'I_weighted_all': np.nan,
            'I_weighted_sig': np.nan
        }
    
    w = lp.weights.lat2W(t_map.shape[0], t_map.shape[1], rook=False)

    filtered_neighbors = {i: [j for j in w.neighbors[i] if j in valid_indices] for i in valid_indices}
    filtered_weights = {i: [w.weights[i][j_idx] for j_idx, j in enumerate(w.neighbors[i]) if j in valid_indices] for i in valid_indices}
    
    # Create a W object from the filtered neighbors and weights
    filtered_w = libpysal_W(filtered_neighbors, filtered_weights)

    # Create the adjacency matrix
    adj_sparse = filtered_w.sparse    
    
    # Find connected components (islands)
    n_components, labels = connected_components(csgraph=adj_sparse, directed=False, return_labels=True)
    
    # Map original indices to contiguous
    original_to_contiguous = {node: idx for idx, node in enumerate(filtered_neighbors.keys())}
    contiguous_to_original = {idx: node for node, idx in original_to_contiguous.items()}
    
    # Remap neighbors and weights to contiguous indices
    remapped_neighbors = {
        original_to_contiguous[node]: [original_to_contiguous[neighbor] for neighbor in neighbors]
        for node, neighbors in filtered_neighbors.items()
    }
    remapped_weights = {
        original_to_contiguous[node]: weights
        for node, weights in filtered_weights.items()
    }
    
    # Get full data
    full_data = t_map_flatten[valid_indices]
    valid_index_map = {original_idx: i for i, original_idx in enumerate(valid_indices)}
    
    # Compute Moran's I per island
    moran_values = []
    p_values = []
    island_details = []
    
    for component_label in tqdm(range(n_components)):
        component_nodes = np.where(labels == component_label)[0]
        
        # Skip small islands (user's reference has min_size=8)
        if len(component_nodes) < min_size:
            continue
        
        # Subset weights for this island
        component_neighbors = {i: remapped_neighbors[i] for i in component_nodes if i in remapped_neighbors}
        component_weights = {i: remapped_weights[i] for i in component_nodes if i in remapped_weights}
        
        if len(component_neighbors) == 0:
            continue
        
        component_w = libpysal_W(component_neighbors, component_weights)
        
        # Map back to data indices
        original_component_nodes = [contiguous_to_original[i] for i in component_nodes if i in contiguous_to_original]
        full_data_indices = [valid_index_map[node] for node in original_component_nodes]
        component_data = full_data[full_data_indices]
        
        # Check validity (user's reference checks for < 2 nodes or islands)
        if len(component_data) < 2 or len(component_w.islands) > 0:
            continue
        
        # Compute Moran's I for this island
        try:
            moran = Moran(component_data, component_w, permutations=n_permutations)
            moran_values.append(moran.I)
            p_values.append(moran.p_sim)
            
            island_details.append({
                'island_id': component_label,
                'n_vertices': len(component_nodes),
                'morans_i': moran.I,
                'p_value': moran.p_sim
            })
            
        except Exception as e:
            print(f"    WARNING: Island {component_label} failed: {e}")
            continue
    
    # Average across islands (user's reference returns average)
    if len(moran_values) > 0:
        avg_moran_i = np.mean(moran_values)
        avg_p_value = np.mean(p_values)
        
        # Extract island sizes for weighted averages
        island_sizes = [detail['n_vertices'] for detail in island_details]
        
        # Weighted average across ALL islands (weighted by island size)
        weighted_sum_all = sum(moran_values[i] * island_sizes[i] for i in range(len(moran_values)))
        total_vertices_all = sum(island_sizes)
        avg_moran_i_weighted_all = weighted_sum_all / total_vertices_all if total_vertices_all > 0 else np.nan
        
        # Compute average across ONLY islands with significant Moran's I (p < 0.05)
        sig_indices = [i for i, p in enumerate(p_values) if p < 0.05]
        if len(sig_indices) > 0:
            sig_moran_values = [moran_values[i] for i in sig_indices]
            avg_moran_i_sig = np.mean(sig_moran_values)
            n_sig_morans_i = len(sig_indices)
            
            # Weighted average across ONLY significant islands
            weighted_sum_sig = sum(moran_values[i] * island_sizes[i] for i in sig_indices)
            total_vertices_sig = sum(island_sizes[i] for i in sig_indices)
            avg_moran_i_weighted_sig = weighted_sum_sig / total_vertices_sig if total_vertices_sig > 0 else np.nan
        else:
            avg_moran_i_sig = np.nan
            avg_moran_i_weighted_sig = np.nan
            n_sig_morans_i = 0
    else:
        avg_moran_i = np.nan
        avg_p_value = np.nan
        avg_moran_i_sig = np.nan
        avg_moran_i_weighted_all = np.nan
        avg_moran_i_weighted_sig = np.nan
        n_sig_morans_i = 0
    
    return {
        'I': avg_moran_i,  # Average across ALL islands (unweighted)
        'p_value': avg_p_value,
        'n_islands': len(moran_values),
        'n_vertices': len(valid_indices),
        'island_details': island_details,
        'I_sig_only': avg_moran_i_sig,  # Average across only significant islands (unweighted)
        'n_sig_morans_i': n_sig_morans_i,  # Count of islands with sig Moran's I
        'I_weighted_all': avg_moran_i_weighted_all,  # NEW: Weighted average across ALL islands
        'I_weighted_sig': avg_moran_i_weighted_sig  # NEW: Weighted average across only sig islands
    }
