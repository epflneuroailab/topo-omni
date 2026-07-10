"""
Motion parameter extraction and outlier detection for Pernet 2015 dataset.

This module implements:
1. Motion parameter extraction from SPM realignment matrices
2. Outlier detection using modified boxplot rule (Carling, 2000)
3. Motion regressor creation for GLM design matrix

Following Pernet et al. (2015) methodology:
- Extract 6 motion parameters (3 translations + 3 rotations) from realignment data
- Identify outlier scans using modified boxplot rule
- Create outlier regressors for design matrix
"""

import numpy as np
import scipy.io
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def extract_motion_parameters_from_mat(mat_file_path: str) -> np.ndarray:
    """
    Extract 6 motion parameters from SPM realignment .mat file.
    
    The .mat file contains 4x4 transformation matrices for each timepoint.
    We extract translation (x, y, z) and rotation (pitch, roll, yaw) parameters.
    
    Parameters
    ----------
    mat_file_path : str
        Path to the .mat file containing realignment matrices
        
    Returns
    -------
    np.ndarray
        Motion parameters array (n_timepoints, 6)
        Columns: [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
    """
    try:
        # Load .mat file
        mat_data = scipy.io.loadmat(mat_file_path)
        
        # Extract transformation matrices (4x4xN)
        if 'mat' in mat_data:
            transform_matrices = mat_data['mat']  # Shape: (4, 4, n_timepoints)
        else:
            raise ValueError(f"No 'mat' key found in {mat_file_path}")
        
        n_timepoints = transform_matrices.shape[2]
        motion_params = np.zeros((n_timepoints, 6))
        
        for t in range(n_timepoints):
            matrix = transform_matrices[:, :, t]
            
            # Extract translations (last column, first 3 elements)
            translations = matrix[:3, 3]
            
            # Extract rotations from rotation matrix (top-left 3x3)
            rotation_matrix = matrix[:3, :3]
            rotations = rotation_matrix_to_euler_angles(rotation_matrix)
            
            # Store motion parameters
            motion_params[t, :3] = translations  # x, y, z translations
            motion_params[t, 3:] = rotations     # pitch, roll, yaw rotations
        
        logger.info(f"Extracted motion parameters from {mat_file_path}")
        logger.info(f"Shape: {motion_params.shape}")
        logger.info(f"Translation ranges: x=[{np.min(motion_params[:, 0]):.3f}, {np.max(motion_params[:, 0]):.3f}], "
                   f"y=[{np.min(motion_params[:, 1]):.3f}, {np.max(motion_params[:, 1]):.3f}], "
                   f"z=[{np.min(motion_params[:, 2]):.3f}, {np.max(motion_params[:, 2]):.3f}]")
        logger.info(f"Rotation ranges: pitch=[{np.min(motion_params[:, 3]):.3f}, {np.max(motion_params[:, 3]):.3f}], "
                   f"roll=[{np.min(motion_params[:, 4]):.3f}, {np.max(motion_params[:, 4]):.3f}], "
                   f"yaw=[{np.min(motion_params[:, 5]):.3f}, {np.max(motion_params[:, 5]):.3f}]")
        
        return motion_params
        
    except Exception as e:
        logger.error(f"Error extracting motion parameters from {mat_file_path}: {e}")
        raise


def rotation_matrix_to_euler_angles(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to Euler angles (pitch, roll, yaw).
    
    Parameters
    ----------
    R : np.ndarray
        3x3 rotation matrix
        
    Returns
    -------
    np.ndarray
        Euler angles [pitch, roll, yaw] in radians
    """
    # Handle identity matrix case
    if np.allclose(R, np.eye(3)):
        return np.zeros(3)
    
    # Extract Euler angles using ZYX convention (yaw, pitch, roll)
    # This matches SPM's convention
    
    # Pitch (rotation around Y-axis)
    sin_pitch = -R[2, 0]
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)  # Clamp to avoid numerical errors
    pitch = np.arcsin(sin_pitch)
    
    # Check for gimbal lock
    if np.abs(np.cos(pitch)) < 1e-6:
        # Gimbal lock case
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        # Normal case
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    
    return np.array([pitch, roll, yaw])


def detect_outliers_carling_2000(data: np.ndarray, k: float = 2.0) -> np.ndarray:
    """
    Detect outliers using modified boxplot rule (Carling, 2000).
    
    This method is more robust than standard boxplot rule for small samples
    and skewed distributions.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (n_samples,) or (n_samples, n_features)
    k : float
        Multiplier for interquartile range (default: 2.0)
        
    Returns
    -------
    np.ndarray
        Boolean array indicating outliers (True = outlier)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    outlier_mask = np.zeros((n_samples, n_features), dtype=bool)
    
    for feature_idx in range(n_features):
        feature_data = data[:, feature_idx]
        
        # Calculate quartiles
        q1 = np.percentile(feature_data, 25)
        q3 = np.percentile(feature_data, 75)
        iqr = q3 - q1
        
        # Modified boxplot rule (Carling, 2000)
        # Uses a more conservative approach for small samples
        if n_samples < 20:
            # For small samples, use more conservative multiplier
            k_adjusted = k * 1.5
        else:
            k_adjusted = k
        
        # Calculate bounds
        lower_bound = q1 - k_adjusted * iqr
        upper_bound = q3 + k_adjusted * iqr
        
        # Identify outliers
        outlier_mask[:, feature_idx] = (feature_data < lower_bound) | (feature_data > upper_bound)
    
    # Return 1D array if input was 1D
    if data.shape[1] == 1:
        return outlier_mask.flatten()
    
    return outlier_mask


def detect_motion_outliers(motion_params: np.ndarray, 
                          global_signal: Optional[np.ndarray] = None,
                          displacement_threshold: float = 2.0,
                          global_signal_threshold: float = 2.0) -> Tuple[np.ndarray, Dict]:
    """
    Detect outlier scans based on motion parameters and global signal.
    
    Following Pernet et al. (2015) and Siegel et al. (2014):
    - Outliers are scans with large mean displacement
    - Outliers are scans with weaker or stronger global signals
    
    Parameters
    ----------
    motion_params : np.ndarray
        Motion parameters (n_timepoints, 6)
    global_signal : Optional[np.ndarray]
        Global signal time series (n_timepoints,)
    displacement_threshold : float
        Threshold for displacement outlier detection (in standard deviations)
    global_signal_threshold : float
        Threshold for global signal outlier detection (in standard deviations)
        
    Returns
    -------
    Tuple[np.ndarray, Dict]
        Boolean outlier mask and outlier statistics
    """
    n_timepoints = motion_params.shape[0]
    outlier_mask = np.zeros(n_timepoints, dtype=bool)
    outlier_stats = {}
    
    # 1. Calculate framewise displacement (FD)
    # FD = sum of absolute derivatives of motion parameters
    # Convert rotations to mm (assuming 50mm head radius)
    head_radius = 50.0  # mm
    motion_params_mm = motion_params.copy()
    motion_params_mm[:, 3:] *= head_radius  # Convert rotations to mm
    
    # Calculate derivatives
    motion_derivatives = np.diff(motion_params_mm, axis=0)
    framewise_displacement = np.sum(np.abs(motion_derivatives), axis=1)
    
    # Pad to match original length (first timepoint has FD = 0)
    fd_full = np.zeros(n_timepoints)
    fd_full[1:] = framewise_displacement
    
    # 2. Detect displacement outliers using Carling (2000) method
    displacement_outliers = detect_outliers_carling_2000(fd_full, k=displacement_threshold)
    outlier_mask |= displacement_outliers
    
    outlier_stats['framewise_displacement'] = fd_full
    outlier_stats['displacement_outliers'] = displacement_outliers
    outlier_stats['n_displacement_outliers'] = np.sum(displacement_outliers)
    
    # 3. Detect global signal outliers (if provided)
    if global_signal is not None:
        global_outliers = detect_outliers_carling_2000(global_signal, k=global_signal_threshold)
        outlier_mask |= global_outliers
        
        outlier_stats['global_signal'] = global_signal
        outlier_stats['global_outliers'] = global_outliers
        outlier_stats['n_global_outliers'] = np.sum(global_outliers)
    
    # 4. Summary statistics
    outlier_stats['total_outliers'] = outlier_mask
    outlier_stats['n_total_outliers'] = np.sum(outlier_mask)
    outlier_stats['outlier_percentage'] = np.sum(outlier_mask) / n_timepoints * 100
    
    logger.info(f"Motion outlier detection results:")
    logger.info(f"  Displacement outliers: {outlier_stats['n_displacement_outliers']}")
    if global_signal is not None:
        logger.info(f"  Global signal outliers: {outlier_stats['n_global_outliers']}")
    logger.info(f"  Total outliers: {outlier_stats['n_total_outliers']} / {n_timepoints} ({outlier_stats['outlier_percentage']:.1f}%)")
    
    return outlier_mask, outlier_stats


def create_motion_regressors(motion_params: np.ndarray, 
                           outlier_mask: np.ndarray,
                           include_derivatives: bool = True) -> Tuple[np.ndarray, list]:
    """
    Create motion regressors for GLM design matrix.
    
    Parameters
    ----------
    motion_params : np.ndarray
        Motion parameters (n_timepoints, 6)
    outlier_mask : np.ndarray
        Boolean outlier mask (n_timepoints,)
    include_derivatives : bool
        Whether to include temporal derivatives of motion parameters
        
    Returns
    -------
    Tuple[np.ndarray, list]
        Motion regressors array and regressor names
    """
    n_timepoints = motion_params.shape[0]
    regressors = []
    regressor_names = []
    
    # 1. Add motion parameters
    regressors.append(motion_params)
    motion_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    regressor_names.extend(motion_names)
    
    # 2. Add motion parameter derivatives (if requested)
    if include_derivatives:
        motion_derivatives = np.zeros_like(motion_params)
        motion_derivatives[1:, :] = np.diff(motion_params, axis=0)
        regressors.append(motion_derivatives)
        derivative_names = [f'{name}_deriv' for name in motion_names]
        regressor_names.extend(derivative_names)
    
    # 3. Add outlier regressors (one regressor per outlier timepoint)
    outlier_indices = np.where(outlier_mask)[0]
    for outlier_idx in outlier_indices:
        outlier_regressor = np.zeros(n_timepoints)
        outlier_regressor[outlier_idx] = 1.0
        regressors.append(outlier_regressor.reshape(-1, 1))
        regressor_names.append(f'outlier_{outlier_idx:03d}')
    
    # Combine all regressors
    motion_regressors = np.concatenate(regressors, axis=1)
    
    logger.info(f"Created motion regressors:")
    logger.info(f"  Motion parameters: 6")
    if include_derivatives:
        logger.info(f"  Motion derivatives: 6")
    logger.info(f"  Outlier regressors: {len(outlier_indices)}")
    logger.info(f"  Total regressors: {motion_regressors.shape[1]}")
    
    return motion_regressors, regressor_names


def extract_motion_from_dataset(subject_id: str, data_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Extract motion parameters and detect outliers for a subject from Pernet 2015 dataset.
    
    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., "sub001_Ed")
    data_dir : str
        Path to dataset directory
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict]
        Motion regressors, outlier mask, and motion statistics
    """
    # Find motion .mat file
    subject_dir = Path(data_dir) / subject_id
    mat_file = subject_dir / "func" / f"{subject_id}.mat"
    
    if not mat_file.exists():
        raise FileNotFoundError(f"Motion .mat file not found: {mat_file}")
    
    # Extract motion parameters
    motion_params = extract_motion_parameters_from_mat(str(mat_file))
    
    # TODO: Extract global signal from functional data for outlier detection
    # For now, we'll just use motion-based outlier detection
    global_signal = None
    
    # Detect outliers
    outlier_mask, outlier_stats = detect_motion_outliers(
        motion_params, 
        global_signal=global_signal
    )
    
    # Create motion regressors
    motion_regressors, regressor_names = create_motion_regressors(
        motion_params, 
        outlier_mask,
        include_derivatives=True
    )
    
    # Add regressor names to stats
    outlier_stats['regressor_names'] = regressor_names
    outlier_stats['motion_params'] = motion_params
    
    return motion_regressors, outlier_mask, outlier_stats 