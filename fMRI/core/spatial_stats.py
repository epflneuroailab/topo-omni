"""Island Moran's I on a cortical surface — the #1 consolidation win (docs/DESIGN.md §4).

Single source of truth for spatial autocorrelation restricted to significant
"islands" (connected supra-threshold vertex clusters). Historically this lived in the
Marvi repo (`src/utils/spatial_stats.py`); Pernet imported it by absolute cross-repo
path and Jung kept a stale duplicate. This module replaces all three.

Used by: Pernet Fig. B3b (island Moran's I of the voice-selective brain map, compared
against the model's value).

PORT NOTES (faithful port of Marvi `src/utils/spatial_stats.py` @ ef1da34):
  * The numeric core (`compute_fdr_threshold`, `compute_island_morans_i`) is copied
    verbatim — it is golden-mastered against Pernet `island_morans_i_results.json`, so
    its arithmetic must not drift.
  * Imports are made LAZY to honor the core import-safety invariant (docs/DESIGN.md §4):
    `import core.spatial_stats` must not pull in nilearn (Pernet 0.10.4 vs Marvi/Jung
    0.12.1). nibabel/nilearn load inside `load_surface_adjacency`; libpysal/esda load
    inside `_require_moran()`. Top level stays numpy/scipy only.
  * Scope-lean (docs/DESIGN.md §2.4): the STANDARD (whole-map) Moran's I path
    (`compute_standard_morans_i`, `adjacency_to_libpysal_W`) is NOT ported — no paper
    figure uses it. Only the island metric ships here.

The permutation p-values (`p_value`, and anything derived from `p < 0.05`:
`n_sig_morans_i`, `I_sig_only`, `I_weighted_sig`) come from esda's unseeded permutation
test and are therefore STOCHASTIC — do not pin them tightly. The Moran's I values
themselves (`I`, `I_weighted_all`, per-island `morans_i`) and the counts (`n_islands`,
`n_vertices`) are deterministic (docs/DESIGN.md §6).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components

# Default parameters (Marvi/model defaults; Pernet passes df/n explicitly).
DEFAULT_N_PERMUTATIONS = 999
DEFAULT_FDR_Q = 0.05
DEFAULT_MIN_ISLAND_SIZE = 8
DEFAULT_N_SUBJECTS = 83
DEFAULT_DF = DEFAULT_N_SUBJECTS - 1


def _require_moran():
    """Lazily import the libpysal/esda Moran's I engine (kept out of module import).

    Returns ``(Moran, W)``. Raises a clear ImportError if the optional stack is
    absent, rather than degrading silently.
    """
    try:
        from esda.moran import Moran
        from libpysal.weights import W as libpysal_W
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "core.spatial_stats needs libpysal + esda for Moran's I. "
            "Install with: pip install libpysal esda"
        ) from exc
    return Moran, libpysal_W


def load_surface_adjacency(hemi=None, fsaverage=None, coords=None, faces=None):
    """Build the vertex-adjacency graph (spatial weights W) for a surface mesh.

    Parameters
    ----------
    hemi : {'L', 'R'}, optional
        Hemisphere. Used to pick the pial surface when fetching via nilearn.
    fsaverage : dict, optional
        Pre-fetched nilearn fsaverage surfaces. If None (and coords/faces not given),
        fsaverage6 is fetched.
    coords, faces : np.ndarray, optional
        Inject a mesh directly (n_vertices×3 coords, n_faces×3 vertex indices),
        bypassing nilearn entirely. Used by unit tests and by callers that already
        hold the mesh — keeps this function usable under either nilearn version.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric adjacency matrix (n_vertices × n_vertices). fsaverage6 → 40,962
        vertices per hemisphere.

    Notes
    -----
    Adjacency is built from the triangular faces: every triangle edge connects two
    adjacent vertices. nibabel/nilearn are imported lazily so that importing this
    module never binds nilearn (docs/DESIGN.md §4).
    """
    if coords is None or faces is None:
        import nibabel as nib  # lazy — see module docstring
        if fsaverage is None:
            from nilearn import datasets
            fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage6')
        surf_key = 'pial_left' if hemi == 'L' else 'pial_right'
        surf = nib.load(fsaverage[surf_key])
        coords = surf.darrays[0].data   # vertex coordinates
        faces = surf.darrays[1].data    # triangular faces

    n_vertices = coords.shape[0]

    # Build adjacency from triangle edges.
    edges = []
    for face in faces:
        edges.append((face[0], face[1]))
        edges.append((face[1], face[2]))
        edges.append((face[2], face[0]))

    edges = np.array(edges)
    edges_unique = np.unique(np.sort(edges, axis=1), axis=0)

    row = np.concatenate([edges_unique[:, 0], edges_unique[:, 1]])
    col = np.concatenate([edges_unique[:, 1], edges_unique[:, 0]])
    data = np.ones(len(row))

    return sparse.csr_matrix((data, (row, col)), shape=(n_vertices, n_vertices))


def compute_fdr_threshold(t_map, p_map, q=DEFAULT_FDR_Q, df=DEFAULT_DF):
    """FDR (Benjamini-Hochberg, one-tailed positive) t-threshold.

    Returns np.inf if no vertex survives. Faithful port — do not alter (golden-master).
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


def _empty_island_result(n_vertices=0):
    return {
        'I': np.nan,
        'p_value': np.nan,
        'n_islands': 0,
        'n_vertices': n_vertices,
        'island_details': [],
        'I_sig_only': np.nan,
        'n_sig_morans_i': 0,
        'I_weighted_all': np.nan,
        'I_weighted_sig': np.nan,
    }


def compute_island_morans_i(t_map, p_map, adj_matrix, fdr_q=DEFAULT_FDR_Q,
                            min_size=DEFAULT_MIN_ISLAND_SIZE,
                            n_permutations=DEFAULT_N_PERMUTATIONS,
                            df=DEFAULT_DF, seed=None):
    """Moran's I computed within FDR-significant islands only.

    1. FDR-threshold the (positive) t-map to get significant vertices.
    2. Restrict the adjacency graph to them and find connected components (islands).
    3. For each island ≥ ``min_size``, compute Moran's I with a permutation test.
    4. Return per-island values plus unweighted/size-weighted averages.

    Returns a dict (see keys below). NaN fields where nothing qualifies.
    Faithful port of the Marvi implementation — the numeric core is golden-mastered
    against Pernet ``island_morans_i_results.json`` and must not be altered.

    Deterministic keys: ``I``, ``I_weighted_all``, ``n_islands``, ``n_vertices``,
    per-island ``morans_i``. Permutation-based keys: ``p_value``, ``I_sig_only``,
    ``n_sig_morans_i``, ``I_weighted_sig``. These are stochastic when ``seed is None``
    (esda 2.5.1's ``Moran`` draws from the global NumPy RNG and takes no seed of its
    own); pass an integer ``seed`` to make them reproducible (docs/DESIGN.md §6). Seeding does
    not touch the deterministic keys.
    """
    Moran, libpysal_W = _require_moran()

    # Get FDR threshold
    fdr_t_threshold = compute_fdr_threshold(t_map, p_map, q=fdr_q, df=df)

    if np.isinf(fdr_t_threshold):
        return _empty_island_result()

    # Mask for FDR-significant vertices (positive only)
    sig_mask = (t_map >= fdr_t_threshold)
    valid_indices = np.where(sig_mask)[0]

    if len(valid_indices) == 0:
        return _empty_island_result()

    # Filter adjacency to significant vertices only
    filtered_neighbors = {}
    filtered_weights = {}

    valid_set = set(valid_indices.tolist())

    for i in valid_indices:
        neighbor_indices = adj_matrix[i].nonzero()[1]
        # Keep only neighbors that are also significant
        filtered_neighbor_list = [j for j in neighbor_indices if j in valid_set]

        if len(filtered_neighbor_list) > 0:
            filtered_neighbors[int(i)] = filtered_neighbor_list
            filtered_weights[int(i)] = [1.0] * len(filtered_neighbor_list)

    if len(filtered_neighbors) == 0:
        return _empty_island_result(n_vertices=len(valid_indices))

    # Create W object
    filtered_w = libpysal_W(filtered_neighbors, filtered_weights)

    # Create adjacency matrix for connected components
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
    full_data = t_map[valid_indices]
    valid_index_map = {original_idx: i for i, original_idx in enumerate(valid_indices)}

    # Seed the global NumPy RNG once before the permutation loop so esda's Moran
    # p-values are reproducible for a fixed input (esda 2.5.1 has no seed kwarg and
    # draws from np.random). Only affects the permutation-based keys, not the
    # deterministic Moran's I values.
    if seed is not None:
        np.random.seed(seed)

    # Compute Moran's I per island
    moran_values = []
    p_values = []
    island_details = []

    for component_label in range(n_components):
        component_nodes = np.where(labels == component_label)[0]

        # Skip small islands (reference has min_size=8)
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

        # Check validity (reference checks for < 2 nodes or islands)
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
                'p_value': moran.p_sim,
            })

        except Exception as e:
            print(f"    WARNING: Island {component_label} failed: {e}")
            continue

    # Average across islands (reference returns average)
    if len(moran_values) > 0:
        avg_moran_i = np.mean(moran_values)
        avg_p_value = np.mean(p_values)

        # Extract island sizes for weighted averages
        island_sizes = [detail['n_vertices'] for detail in island_details]

        # Weighted average across ALL islands (weighted by island size)
        weighted_sum_all = sum(moran_values[i] * island_sizes[i] for i in range(len(moran_values)))
        total_vertices_all = sum(island_sizes)
        avg_moran_i_weighted_all = weighted_sum_all / total_vertices_all if total_vertices_all > 0 else np.nan

        # Average across ONLY islands with significant Moran's I (p < 0.05)
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
        'I_weighted_all': avg_moran_i_weighted_all,  # Weighted average across ALL islands
        'I_weighted_sig': avg_moran_i_weighted_sig,  # Weighted average across only sig islands
    }
