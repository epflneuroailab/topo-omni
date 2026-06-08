import numpy as np
import libpysal as lp
from esda.moran import Moran
from libpysal.weights import W as libpysal_W
from libpysal.weights import remap_ids
from scipy.sparse.csgraph import connected_components

def island_morans_I(p_map, t_map, p_threshold=0.05) -> float:
    """
    Moran's I computation that handles islands (nodes with no neighbors)
    by computing Moran's I separately for each connected component.
    """
    p_map_masked = (p_map < p_threshold) & (t_map > 0)
    p_map_masked_flat = p_map_masked.flatten()
    w = lp.weights.lat2W(t_map.shape[0], t_map.shape[1], rook=False)
    valid_indices = np.where(p_map_masked_flat)[0]

    #filtered_weights = w.subset(sig_units_indices)
    filtered_neighbors = {i: [j for j in w.neighbors[i] if j in valid_indices] for i in valid_indices}
    filtered_weights = {i: [w.weights[i][j_idx] for j_idx, j in enumerate(w.neighbors[i]) if j in valid_indices] for i in valid_indices}
    
    # Create a W object from the filtered neighbors and weights
    filtered_w = libpysal_W(filtered_neighbors, filtered_weights)

    # Create the adjacency matrix
    adj_matrix = filtered_w.sparse

    # Find connected components
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)

    # Map original node indices to contiguous ones
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
    remapped_w = libpysal_W(remapped_neighbors, remapped_weights)

    # Prepare data for all components
    full_data = t_map.flatten()[p_map_masked_flat]

    # Map valid indices to positions in `full_data`
    valid_index_map = {original_idx: i for i, original_idx in enumerate(valid_indices)}

    moran_values = []
    island_moran_values = {}
    num_significant_components = 0
        
    for component_id, component_label in enumerate(range(n_components)):
        # Get contiguous indices for the current component
        component_nodes = np.where(labels == component_label)[0]
        
        # Skip components with fewer than 8 nodes (islands with no neighbors)
        if len(component_nodes) < 8:
            print(f"Skipping component {component_label} with {len(component_nodes)} node(s)")
            continue
        
        # Subset the remapped weights for the current component
        component_neighbors = {i: remapped_neighbors[i] for i in component_nodes}
        component_weights = {i: remapped_weights[i] for i in component_nodes}
        component_w = libpysal_W(component_neighbors, component_weights)
        
        # Map component nodes back to positions in `full_data`
        original_component_nodes = [contiguous_to_original[i] for i in component_nodes]
        full_data_indices = [valid_index_map[node] for node in original_component_nodes]
        
        # Subset the data for the current component
        component_data = full_data[full_data_indices]
        
        # Check if the component weights matrix is valid
        if len(component_data) < 2 or len(component_w.islands) > 0:
            print(f"Skipping invalid component {component_label}")
            continue

        # Compute Moran's I
        moran = Moran(component_data, component_w, permutations=999)
        moran_values.append(moran.I)
        if moran.p_sim < p_threshold:
            num_significant_components += 1

        island_moran_values[component_id] = dict(
            moran_I=moran.I,
            p_value=moran.p_sim,
        )

        # print
        print(f"Component {component_label} with {len(component_nodes)} nodes: Moran's I = {moran.I:.3f} | p-value = {moran.p_sim:.3f}")

    # Average Moran's I across all components
    average_moran_I = np.mean(moran_values) if moran_values else np.nan
    # print(f"Average Moran's I across components: {average_moran_I:.3f} | Average p-value = {average_p_value:.3f}")
    return dict(
        average_moran_I=average_moran_I,
        num_components=n_components,
        num_significant_components=num_significant_components,
        island_moran_values=island_moran_values
    )