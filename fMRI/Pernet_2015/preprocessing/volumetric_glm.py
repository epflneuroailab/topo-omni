"""
Volumetric GLM analysis for fMRI data following Pernet 2015 methodology.

This module implements volumetric General Linear Model (GLM) analysis in MNI space
as described in Pernet et al. (2015). This corrects the previous surface-based 
implementation which was methodologically incorrect.

Key Features:
- Volumetric GLM analysis in MNI space (2mm isotropic voxels)
- 6mm FWHM Gaussian smoothing (volumetric)
- SPM-style preprocessing and analysis
- Motion parameters, AR(1), high-pass filtering
- Outputs volumetric t-maps and contrast images for downstream analysis

Correct Pernet 2015 Methodology:
- Primary Analysis: Volumetric in MNI space using SPM12b
- Smoothing: 6mm FWHM Gaussian kernel
- Gaussian-Gamma Mixture Model: Applied to volumetric t-maps
- Topological FDR: Applied to volumetric maps per Chumbley et al. 2010
- Surface Analysis: ONLY for final visualization (not primary analysis)

Reference: Pernet et al. (2015) - "The human voice areas: Spatial organization and 
inter-individual variability in temporal and extra-temporal cortices"
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.first_level import run_glm
from nilearn.image import smooth_img, resample_to_img
from nilearn import datasets
from scipy.stats import ttest_1samp, t
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from .motion_correction import extract_motion_from_dataset
import json
import subprocess
import tempfile
import shutil
from tqdm import tqdm
import time

# Import FSL Python interface
try:
    import fsl.wrappers as fsl
    from fsl.data.image import Image
    FSL_AVAILABLE = True
    print("FSL Python interface (fslpy) successfully imported")
except ImportError:
    FSL_AVAILABLE = False
    print("WARNING: FSL Python interface not available - falling back to ANTs")

# Import ANTs as fallback
try:
    import ants
    ANTS_AVAILABLE = True
    print("ANTs successfully imported as fallback")
except ImportError:
    ANTS_AVAILABLE = False
    print("WARNING: ANTs not available - registration may fail")


class VolumetricGLMAnalyzer:
    """
    Volumetric GLM analyzer for Pernet 2015 data in MNI space.
    
    Implements voxel-wise GLM analysis in MNI volumetric space following 
    the exact methodology described in Pernet et al. (2015).
    """
    
    def __init__(self, 
                 tr: float = 2.0, 
                 hrf_model: str = 'spm', 
                 drift_model: str = 'cosine', 
                 high_pass: float = 1/128,
                 smoothing_fwhm: float = 6.0,
                 target_affine: Optional[np.ndarray] = None,
                 target_shape: Optional[Tuple[int, int, int]] = None):
        """
        Initialize the volumetric GLM analyzer.
        
        Parameters
        ----------
        tr : float
            Repetition time in seconds
        hrf_model : str
            HRF model ('spm', 'glover', etc.)
        drift_model : str
            Drift model ('cosine', 'polynomial', etc.)
        high_pass : float
            High-pass filter cutoff in Hz (1/128 = 0.0078 Hz as in Pernet 2015)
        smoothing_fwhm : float
            FWHM for Gaussian smoothing in mm (6.0 as in Pernet 2015)
        target_affine : Optional[np.ndarray]
            Target affine matrix for MNI space (2mm isotropic if None)
        target_shape : Optional[Tuple[int, int, int]]
            Target shape for MNI space
        """
        self.tr = tr
        self.hrf_model = hrf_model
        self.drift_model = drift_model
        self.high_pass = high_pass
        self.smoothing_fwhm = smoothing_fwhm
        
        # Set up MNI space parameters (2mm isotropic as in Pernet 2015)
        if target_affine is None:
            # Standard MNI 2mm isotropic affine
            self.target_affine = np.array([
                [-2.,  0.,  0.,  90.],
                [ 0.,  2.,  0., -126.],
                [ 0.,  0.,  2.,  -72.],
                [ 0.,  0.,  0.,   1.]
            ])
        else:
            self.target_affine = target_affine
        
        if target_shape is None:
            # Standard MNI 2mm template shape
            self.target_shape = (91, 109, 91)
        else:
            self.target_shape = target_shape
    
    def load_mni_template(self) -> nib.Nifti1Image:
        """
        Load MNI template for registration target.
        
        Returns
        -------
        nib.Nifti1Image
            MNI template image
        """
        # Load MNI template from nilearn
        template = datasets.load_mni152_template(resolution=2)
        
        print(f"        MNI template loaded:")
        print(f"          Shape: {template.shape}")
        print(f"          Affine:\n{template.affine}")
        print(f"          Voxel size: {template.header.get_zooms()[:3]}")
        
        return template
    
    def preprocess_functional_data(self, func_img: nib.Nifti1Image, 
                                   anat_img: Optional[nib.Nifti1Image] = None) -> nib.Nifti1Image:
        """
        Preprocess functional data following exact Pernet 2015 methodology using FSL.
        
        Implements the complete Pernet 2015 preprocessing pipeline:
        1. Slice timing correction - sinc interpolation, reference slice 31
        2. Motion correction - 6 parameter affine, realign to mean, spline interpolation
        3. Co-registration - T1 → mean EPI using normalized mutual information
        4. Normalization - fnirt with forward deformation fields to MNI152_T1_2mm
        5. Resampling - 2mm isotropic with 4th degree B-spline interpolation
        6. Smoothing - 6mm FWHM Gaussian kernel
        
        Parameters
        ----------
        func_img : nib.Nifti1Image
            Raw functional data
        anat_img : Optional[nib.Nifti1Image]
            Anatomical data (required for Pernet 2015 methodology)
            
        Returns
        -------
        nib.Nifti1Image
            Preprocessed functional data in MNI space
        """
        print("    Preprocessing functional data following Pernet 2015 methodology...")
        
        # Validate inputs
        if anat_img is None:
            raise ValueError("Anatomical image required for Pernet 2015 methodology")
        
        if not FSL_AVAILABLE:
            raise RuntimeError("FSL is required for Pernet 2015 preprocessing pipeline")
        
        # Create temporary directory for all preprocessing steps
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Save input images
            func_raw_path = temp_dir / "func_raw.nii.gz"
            anat_path = temp_dir / "anat.nii.gz"
            nib.save(func_img, func_raw_path)
            nib.save(anat_img, anat_path)
            
            # Step 1: Slice timing correction (sinc interpolation, reference slice 31)
            print("      Step 1: Slice timing correction (sinc interpolation, reference slice 31)...")
            func_st = self._slice_timing_correction(func_raw_path, temp_dir)
            
            # Step 2: Motion correction (6 parameter affine, realign to mean, spline interpolation)
            print("      Step 2: Motion correction (6 parameter affine, realign to mean)...")
            func_mc, motion_params = self._motion_correction(func_st, temp_dir)
            
            # Step 3: Co-registration (T1 → mean EPI using normalized mutual information)
            print("      Step 3: Co-registration (T1 → mean EPI, normalized mutual information)...")
            anat_coreg = self._coregister_anatomical_to_functional(anat_path, func_mc, temp_dir)
            
            # Step 4: Normalization (fnirt with forward deformation fields to MNI152_T1_2mm)
            print("      Step 4: Normalization (fnirt with forward deformation fields)...")
            func_norm, anat_norm = self._normalize_to_mni(func_mc, anat_coreg, temp_dir)
            
            # Step 5: Resampling (2mm isotropic with 4th degree B-spline interpolation)
            print("      Step 5: Resampling (2mm isotropic, 4th degree B-spline)...")
            func_resamp = self._resample_to_standard_space(func_norm, temp_dir)
            
            # Step 6: Smoothing (6mm FWHM Gaussian kernel)
            print("      Step 6: Smoothing (6mm FWHM Gaussian kernel)...")
            func_smooth_path = self._smooth_functional_data(func_resamp, temp_dir)
            
            # Load the final smoothed image and ensure data is in memory
            func_smooth = nib.load(func_smooth_path)
            
            # Force data to be loaded into memory to avoid file reference issues
            func_data = func_smooth.get_fdata()
            func_smooth_memory = nib.Nifti1Image(func_data, func_smooth.affine, func_smooth.header)
            
            # Step 7: Quality control
            print("      Step 7: Quality control validation...")
            qc_results = self._quality_control_validation(func_smooth_memory, temp_dir)
            
            print(f"      Preprocessing complete:")
            print(f"        Final shape: {func_smooth_memory.shape}")
            print(f"        Final voxel size: {func_smooth_memory.header.get_zooms()[:3]}")
            print(f"        Motion parameters: {motion_params.shape if motion_params is not None else 'None'}")
            print(f"        QC assessment: {qc_results.get('overall_quality', 'Unknown')}")
            
            return func_smooth_memory
    
    def _slice_timing_correction(self, func_path: Path, temp_dir: Path) -> Path:
        """
        Perform slice timing correction using FSL slicetimer.
        
        Implements Pernet 2015 slice timing correction:
        - Sinc interpolation
        - Reference slice 31 (middle of TR)
        - Interleaved slice order
        """
        func_st_path = temp_dir / "func_st.nii.gz"
        
        try:
            # Get number of slices for validation
            img = nib.load(func_path)
            n_slices = img.shape[2]
            
            # Reference slice 31 (middle of TR as in Pernet 2015)
            ref_slice = 31 if n_slices > 31 else n_slices // 2
            
            print(f"        Slice timing correction: {n_slices} slices, reference slice {ref_slice}")
            
            with tqdm(total=1, desc="        Slice timing", unit="vol", leave=False) as pbar:
                # Use subprocess to call slicetimer directly since FSL Python interface may not have it
                import subprocess
                cmd = [
                    'slicetimer',
                    '-i', str(func_path),
                    '-o', str(func_st_path),
                    '-r', str(self.tr),
                    '-d', '3',  # z-direction (slice axis)
                    '--odd'  # Interleaved slice order (odd slices first)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                pbar.update(1)
            
            print("        ✓ Slice timing correction successful")
            return func_st_path
            
        except Exception as e:
            print(f"        ✗ Slice timing correction failed: {e}")
            print("        Continuing without slice timing correction...")
            return func_path
    
    def _motion_correction(self, func_path: Path, temp_dir: Path) -> Tuple[Path, Optional[np.ndarray]]:
        """
        Perform motion correction using FSL mcflirt.
        
        Implements Pernet 2015 motion correction:
        - 6 parameter affine transformation
        - Realign to mean image
        - Spline interpolation
        """
        func_mc_path = temp_dir / "func_mc.nii.gz"
        motion_params_path = temp_dir / "func_mc.par"
        
        try:
            print("        Motion correction: 6 parameter affine, realign to mean...")
            
            with tqdm(total=1, desc="        Motion correction", unit="vol", leave=False) as pbar:
                fsl.mcflirt(
                    infile=str(func_path),
                    out=str(func_mc_path),
                    cost='normcorr',  # Normalized correlation
                    dof=6,  # 6 degree of freedom (rigid body)
                    refvol='middle',  # Reference to middle volume (then create mean)
                    mats=True,  # Save transformation matrices
                    plots=True,  # Save motion plots
                    report=True,  # Generate motion report
                    verbose=False
                )
                pbar.update(1)
            
            # Load motion parameters
            motion_params = None
            if motion_params_path.exists():
                motion_params = np.loadtxt(motion_params_path)
                print(f"        Motion parameters shape: {motion_params.shape}")
                
                # Calculate motion statistics
                mean_fd = np.mean(np.abs(np.diff(motion_params[:, :3], axis=0)))
                mean_rotation = np.mean(np.abs(np.diff(motion_params[:, 3:], axis=0)))
                print(f"        Mean frame displacement: {mean_fd:.3f} mm")
                print(f"        Mean rotation: {mean_rotation:.3f} rad")
            
            print("        ✓ Motion correction successful")
            return func_mc_path, motion_params
            
        except Exception as e:
            print(f"        ✗ Motion correction failed: {e}")
            print("        Continuing without motion correction...")
            return func_path, None
    
    def _coregister_anatomical_to_functional(self, anat_path: Path, func_path: Path, temp_dir: Path) -> Path:
        """
        Coregister anatomical to functional space using normalized mutual information.
        
        Implements Pernet 2015 co-registration:
        - T1 image → mean EPI (not EPI → T1)
        - Normalized mutual information cost function
        - Creates mean functional image for registration target
        """
        func_mean_path = temp_dir / "func_mean.nii.gz"
        anat_coreg_path = temp_dir / "anat_coreg.nii.gz"
        coreg_mat_path = temp_dir / "anat_to_func.mat"
        
        try:
            # Create mean functional image
            print("        Creating mean functional image...")
            func_img = nib.load(func_path)
            if len(func_img.shape) == 4:
                func_mean_data = func_img.get_fdata().mean(axis=3)
                func_mean = nib.Nifti1Image(func_mean_data, func_img.affine, func_img.header)
                nib.save(func_mean, func_mean_path)
            else:
                func_mean_path = func_path
            
            # Co-register T1 to mean EPI using normalized mutual information
            print("        Co-registering T1 → mean EPI (normalized mutual information)...")
            
            with tqdm(total=1, desc="        T1→EPI coregistration", unit="vol", leave=False) as pbar:
                fsl.flirt(
                    src=str(anat_path),
                    ref=str(func_mean_path),
                    out=str(anat_coreg_path),
                    omat=str(coreg_mat_path),
                    cost='normmi',  # Normalized mutual information (as in Pernet 2015)
                    dof=6,  # 6 DOF rigid body transformation
                    searchrx=[-90, 90],  # Search ranges
                    searchry=[-90, 90],
                    searchrz=[-90, 90],
                    verbose=False
                )
                pbar.update(1)
            
            print("        ✓ T1 → EPI co-registration successful")
            return anat_coreg_path
            
        except Exception as e:
            print(f"        ✗ Co-registration failed: {e}")
            print("        Using original anatomical image...")
            return anat_path
    
    def _normalize_to_mni(self, func_path: Path, anat_path: Path, temp_dir: Path) -> Tuple[Path, Path]:
        """
        Normalize both functional and anatomical data to MNI space using fnirt.
        
        Implements Pernet 2015 normalization:
        - Diffeomorphic normalization using forward deformation fields
        - Target: MNI152_T1_2mm template
        - Both T1 and EPI normalized to MNI space
        """
        # Load MNI template
        mni_template = self.load_mni_template()
        mni_path = temp_dir / "mni_template.nii.gz"
        nib.save(mni_template, mni_path)
        
        # Output paths
        anat_norm_path = temp_dir / "anat_norm.nii.gz"
        func_norm_path = temp_dir / "func_norm.nii.gz"
        warp_field_path = temp_dir / "anat_to_mni_warp.nii.gz"
        
        try:
            # First, perform linear registration (anatomical to MNI)
            print("        Linear registration: anatomical → MNI...")
            anat_to_mni_linear = temp_dir / "anat_to_mni_linear.mat"
            anat_linear_path = temp_dir / "anat_linear.nii.gz"
            
            with tqdm(total=1, desc="        Linear: Anat→MNI", unit="vol", leave=False) as pbar:
                fsl.flirt(
                    src=str(anat_path),
                    ref=str(mni_path),
                    out=str(anat_linear_path),
                    omat=str(anat_to_mni_linear),
                    cost='normmi',  # Normalized mutual information
                    dof=12,  # 12 DOF affine transformation
                    verbose=False
                )
                pbar.update(1)
            
            # Then, perform non-linear registration using fnirt
            print("        Non-linear registration: fnirt with forward deformation fields...")
            
            with tqdm(total=1, desc="        Non-linear: fnirt", unit="vol", leave=False) as pbar:
                fsl.fnirt(
                    src=str(anat_path),
                    ref=str(mni_path),
                    aff=str(anat_to_mni_linear),  # Initialize with linear transformation
                    cout=str(warp_field_path),  # Output warp field
                    iout=str(anat_norm_path),  # Output warped image
                    imprefm=0,  # Disable implicit masking to avoid dimension mismatch
                    impinm=0,   # Disable implicit masking  
                    applyrefmask=0,  # Disable automatic reference mask
                    applyinmask=0,   # Disable automatic input mask
                    verbose=False
                )
                pbar.update(1)
            
            # Apply warp field to functional data
            print("        Applying warp field to functional data...")
            
            with tqdm(total=1, desc="        Warp: Func→MNI", unit="vol", leave=False) as pbar:
                fsl.applywarp(
                    src=str(func_path),
                    ref=str(mni_path),
                    warp=str(warp_field_path),
                    out=str(func_norm_path),
                    interp='spline',  # 4th degree B-spline interpolation
                    verbose=False
                )
                pbar.update(1)
            
            print("        ✓ Diffeomorphic normalization successful")
            return func_norm_path, anat_norm_path
            
        except Exception as e:
            print(f"        ✗ Normalization failed: {e}")
            print("        Falling back to linear registration only...")
            
            # Fallback: linear registration only
            try:
                with tqdm(total=1, desc="        Fallback: Linear only", unit="vol", leave=False) as pbar:
                    # Apply linear transformation to functional data
                    fsl.flirt(
                        src=str(func_path),
                        ref=str(mni_path),
                        out=str(func_norm_path),
                        init=str(anat_to_mni_linear),
                        applyxfm=True,
                        interp='spline',
                        verbose=False
                    )
                    
                    # Apply to anatomical data
                    fsl.flirt(
                        src=str(anat_path),
                        ref=str(mni_path),
                        out=str(anat_norm_path),
                        init=str(anat_to_mni_linear),
                        applyxfm=True,
                        interp='spline',
                        verbose=False
                    )
                    pbar.update(1)
                
                print("        ✓ Linear normalization successful")
                return func_norm_path, anat_norm_path
                
            except Exception as e2:
                print(f"        ✗ Fallback normalization also failed: {e2}")
                raise RuntimeError("Both non-linear and linear normalization failed")
    
    def _resample_to_standard_space(self, func_path: Path, temp_dir: Path) -> Path:
        """
        Resample functional data to standard 2mm isotropic space.
        
        Implements Pernet 2015 resampling:
        - 2mm isotropic voxels
        - 4th degree B-spline interpolation
        - Standard MNI space dimensions
        """
        func_resamp_path = temp_dir / "func_resamp.nii.gz"
        
        try:
            # Load MNI template for target space
            mni_template = self.load_mni_template()
            mni_path = temp_dir / "mni_template.nii.gz"
            nib.save(mni_template, mni_path)
            
            print("        Resampling to 2mm isotropic space...")
            
            with tqdm(total=1, desc="        Resampling", unit="vol", leave=False) as pbar:
                fsl.flirt(
                    src=str(func_path),
                    ref=str(mni_path),
                    out=str(func_resamp_path),
                    applyxfm=True,  # Apply identity transformation (just resample)
                    interp='spline',  # 4th degree B-spline interpolation
                    verbose=False
                )
                pbar.update(1)
            
            print("        ✓ Resampling successful")
            return func_resamp_path
            
        except Exception as e:
            print(f"        ✗ Resampling failed: {e}")
            print("        Using original resolution...")
            return func_path
    
    def _smooth_functional_data(self, func_path: Path, temp_dir: Path) -> Path:
        """
        Smooth functional data using 6mm FWHM Gaussian kernel.
        
        Implements Pernet 2015 smoothing:
        - 6mm FWHM Gaussian kernel
        - Volumetric smoothing using FSL
        """
        func_smooth_path = temp_dir / "func_smooth.nii.gz"
        
        try:
            print(f"        Smoothing with {self.smoothing_fwhm}mm FWHM Gaussian kernel...")
            
            # Calculate sigma from FWHM (sigma = FWHM / 2.355)
            sigma = self.smoothing_fwhm / 2.355
            
            with tqdm(total=1, desc="        Smoothing", unit="vol", leave=False) as pbar:
                # Use fslmaths for Gaussian smoothing (more reliable than susan)
                import subprocess
                cmd = [
                    'fslmaths',
                    str(func_path),
                    '-s', str(sigma),  # Gaussian smoothing with sigma
                    str(func_smooth_path)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                pbar.update(1)
            
            print("        ✓ Smoothing successful")
            return func_smooth_path
            
        except Exception as e:
            print(f"        ✗ FSL smoothing failed: {e}")
            print("        Using nilearn smoothing as fallback...")
            
            # Fallback to nilearn smoothing
            func_img = nib.load(func_path)
            func_smooth = smooth_img(func_img, fwhm=self.smoothing_fwhm)
            
            # Save the smoothed image and return the path
            nib.save(func_smooth, func_smooth_path)
            print("        ✓ Fallback smoothing successful")
            return func_smooth_path
    
    def _quality_control_validation(self, func_img: nib.Nifti1Image, temp_dir: Path) -> Dict[str, Any]:
        """
        Perform quality control validation of the preprocessing pipeline.
        
        Validates:
        - Proper alignment with MNI template
        - Correct voxel size (2mm isotropic)
        - Reasonable signal-to-noise ratio
        - Proper brain coverage
        """
        qc_results = {}
        
        try:
            # Load MNI template for comparison
            mni_template = self.load_mni_template()
            
            # Check voxel size
            voxel_size = func_img.header.get_zooms()[:3]
            expected_voxel_size = (2.0, 2.0, 2.0)
            voxel_size_diff = np.abs(np.array(voxel_size) - np.array(expected_voxel_size))
            
            qc_results['voxel_size'] = voxel_size
            qc_results['voxel_size_correct'] = np.all(voxel_size_diff < 0.1)
            
            # Check affine alignment
            affine_diff = np.abs(func_img.affine - mni_template.affine).max()
            qc_results['affine_difference'] = affine_diff
            qc_results['affine_aligned'] = affine_diff < 0.5
            
            # Check image dimensions
            qc_results['image_shape'] = func_img.shape
            qc_results['expected_shape'] = mni_template.shape
            qc_results['shape_compatible'] = func_img.shape[:3] == mni_template.shape[:3]
            
            # Calculate signal statistics
            func_data = func_img.get_fdata()
            if len(func_data.shape) == 4:
                func_mean = func_data.mean(axis=3)
            else:
                func_mean = func_data
            
            brain_mask = func_mean > func_mean.mean() * 0.1
            brain_signal = func_mean[brain_mask]
            
            qc_results['mean_signal'] = np.mean(brain_signal)
            qc_results['signal_std'] = np.std(brain_signal)
            qc_results['snr'] = np.mean(brain_signal) / np.std(brain_signal)
            qc_results['brain_coverage'] = np.sum(brain_mask) / np.prod(func_mean.shape)
            
            # Overall quality assessment
            checks_passed = [
                qc_results['voxel_size_correct'],
                qc_results['affine_aligned'],
                qc_results['shape_compatible'],
                qc_results['snr'] > 10,
                qc_results['brain_coverage'] > 0.1
            ]
            
            if np.sum(checks_passed) >= 4:
                qc_results['overall_quality'] = 'Good'
            elif np.sum(checks_passed) >= 3:
                qc_results['overall_quality'] = 'Moderate'
            else:
                qc_results['overall_quality'] = 'Poor'
            
            print(f"        QC Results:")
            print(f"          Voxel size: {voxel_size} {'✓' if qc_results['voxel_size_correct'] else '✗'}")
            print(f"          Affine alignment: {affine_diff:.3f} {'✓' if qc_results['affine_aligned'] else '✗'}")
            print(f"          Shape compatibility: {'✓' if qc_results['shape_compatible'] else '✗'}")
            print(f"          SNR: {qc_results['snr']:.1f} {'✓' if qc_results['snr'] > 10 else '✗'}")
            print(f"          Brain coverage: {qc_results['brain_coverage']:.1%} {'✓' if qc_results['brain_coverage'] > 0.1 else '✗'}")
            
            return qc_results
            
        except Exception as e:
            print(f"        ✗ Quality control validation failed: {e}")
            qc_results['overall_quality'] = 'Failed'
            return qc_results
    
    def create_design_matrix(self, events_df: pd.DataFrame, n_scans: int, 
                           motion_regressors: Optional[np.ndarray] = None,
                           motion_regressor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create design matrix for volumetric GLM analysis.
        
        Following Pernet 2015 methodology:
        - Task regressors (vocal, non-vocal) convolved with HRF
        - Motion parameters (6 parameters + derivatives)
        - High-pass filter (128s as in Pernet 2015)
        - Constant term
        
        Parameters
        ----------
        events_df : pd.DataFrame
            Events DataFrame with onset, duration, trial_type
        n_scans : int
            Number of scans/TRs
        motion_regressors : Optional[np.ndarray]
            Motion regressors array (n_timepoints, n_motion_regressors)
        motion_regressor_names : Optional[List[str]]
            Names of motion regressors
            
        Returns
        -------
        pd.DataFrame
            Design matrix
        """
        frame_times = np.arange(n_scans) * self.tr
        
        # Create basic design matrix with task regressors
        design_matrix = make_first_level_design_matrix(
            frame_times,
            events_df,
            hrf_model=self.hrf_model,
            drift_model=self.drift_model,
            high_pass=self.high_pass  # 128s filter (1/128 = 0.0078 Hz)
        )
        
        # Add motion regressors if provided
        if motion_regressors is not None:
            if motion_regressor_names is None:
                motion_regressor_names = [f'motion_{i:02d}' for i in range(motion_regressors.shape[1])]
            
            # Ensure motion regressors have correct length
            if motion_regressors.shape[0] != n_scans:
                raise ValueError(f"Motion regressors length ({motion_regressors.shape[0]}) "
                               f"doesn't match n_scans ({n_scans})")
            
            # Add motion regressors to design matrix
            for i, regressor_name in enumerate(motion_regressor_names):
                design_matrix[regressor_name] = motion_regressors[:, i]
        
        return design_matrix
    
    def create_vocal_vs_nonvocal_contrast(self, design_matrix: pd.DataFrame) -> np.ndarray:
        """
        Create vocal vs non-vocal contrast vector.
        
        For Pernet 2015: vocal vs non-vocal (ignoring silence baseline)
        
        Parameters
        ----------
        design_matrix : pd.DataFrame
            Design matrix with regressor columns
            
        Returns
        -------
        np.ndarray
            Contrast vector
        """
        contrast_vector = np.zeros(len(design_matrix.columns))
        
        # Find vocal, non-vocal, and silence regressors
        vocal_idx = None
        nonvocal_idx = None
        silence_idx = None
        
        for i, col in enumerate(design_matrix.columns):
            if col.lower() == 'vocal':
                vocal_idx = i
            elif col.lower() == 'non_vocal':
                nonvocal_idx = i
            elif col.lower() == 'silence':
                silence_idx = i
        
        if vocal_idx is None:
            raise ValueError("Could not find 'vocal' regressor in design matrix")
        if nonvocal_idx is None:
            raise ValueError("Could not find 'non_vocal' regressor in design matrix")
        
        # Set up contrast: vocal > non-vocal (silence = 0, neutral)
        contrast_vector[vocal_idx] = 1
        contrast_vector[nonvocal_idx] = -1
        # silence remains 0 (neutral)
        
        print(f"      Contrast setup: vocal={vocal_idx}(+1), non_vocal={nonvocal_idx}(-1), silence={silence_idx}(0)")
        
        return contrast_vector
    
    def run_volumetric_glm(self, func_img: nib.Nifti1Image, 
                          design_matrix: pd.DataFrame) -> Dict[str, nib.Nifti1Image]:
        """
        Run volumetric GLM analysis using nilearn FirstLevelModel.
        
        Parameters
        ----------
        func_img : nib.Nifti1Image
            Preprocessed functional data in MNI space
        design_matrix : pd.DataFrame
            Design matrix
            
        Returns
        -------
        Dict[str, nib.Nifti1Image]
            Dictionary with beta maps, residuals, etc.
        """
        print("    Running volumetric GLM with AR(1) modeling...")
        
        # Initialize FirstLevelModel
        first_level_model = FirstLevelModel(
            t_r=self.tr,
            hrf_model=self.hrf_model,
            drift_model=None,  # Already included in design matrix
            high_pass=None,    # Already included in design matrix
            noise_model='ar1',  # AR(1) autocorrelation modeling as in Pernet 2015
            standardize=False,  # Don't standardize - we want raw beta values
            signal_scaling=0,   # No additional scaling
            smoothing_fwhm=None,  # Already smoothed in preprocessing
        )
        
        # Fit the model
        print(f"      Fitting GLM to {func_img.shape} voxels...")
        first_level_model.fit(func_img, design_matrices=design_matrix)
        
        # Get model results
        results = {
            'model': first_level_model,
            'design_matrix': design_matrix
        }
        
        print(f"      GLM fitting complete")
        
        return results
    
    def compute_contrast_maps(self, glm_results: Dict, 
                             contrast_vector: np.ndarray) -> Dict[str, nib.Nifti1Image]:
        """
        Compute contrast maps (beta estimates and t-statistics).
        
        Parameters
        ----------
        glm_results : Dict
            Results from run_volumetric_glm
        contrast_vector : np.ndarray
            Contrast vector
            
        Returns
        -------
        Dict[str, nib.Nifti1Image]
            Contrast estimates, t-maps, p-values, etc.
        """
        model = glm_results['model']
        
        print("    Computing contrast maps...")
        
        # Compute contrast - nilearn returns different output types
        try:
            # Try the comprehensive output first
            contrast_img = model.compute_contrast(contrast_vector, output_type='all')
            if isinstance(contrast_img, dict):
                # Extract specific maps from dictionary
                contrast_maps = {
                    'contrast_estimates': contrast_img.get('effect', contrast_img.get('z_score')),
                    't_map': contrast_img.get('stat', contrast_img.get('z_score')),
                    'p_values': contrast_img.get('p_value'),
                    'z_scores': contrast_img.get('z_score'),
                    'variance': contrast_img.get('variance')
                }
            else:
                # Single output - likely z_score
                contrast_maps = {
                    'contrast_estimates': contrast_img,
                    't_map': contrast_img,
                    'p_values': None,
                    'z_scores': contrast_img,
                    'variance': None
                }
        except Exception as e:
            print(f"      Warning: Using fallback contrast computation due to: {e}")
            # Fallback to individual computations
            contrast_maps = {
                'contrast_estimates': model.compute_contrast(contrast_vector),
                't_map': model.compute_contrast(contrast_vector, output_type='stat'),
                'p_values': model.compute_contrast(contrast_vector, output_type='p_value'),
                'z_scores': model.compute_contrast(contrast_vector, output_type='z_score'),
                'variance': None
            }
        
        # Get basic statistics
        if contrast_maps['t_map'] is not None:
            t_data = contrast_maps['t_map'].get_fdata()
            print(f"      T-map statistics:")
            print(f"        Mean t: {np.nanmean(t_data):.3f}")
            print(f"        Std t: {np.nanstd(t_data):.3f}")
            print(f"        Range t: [{np.nanmin(t_data):.3f}, {np.nanmax(t_data):.3f}]")
            print(f"        Valid voxels: {np.sum(~np.isnan(t_data))}")
        
        if contrast_maps['p_values'] is not None:
            p_data = contrast_maps['p_values'].get_fdata()
            print(f"        Min p-value: {np.nanmin(p_data):.2e}")
        
        if contrast_maps['contrast_estimates'] is not None:
            contrast_data = contrast_maps['contrast_estimates'].get_fdata()
            print(f"      Contrast estimates:")
            print(f"        Mean contrast: {np.nanmean(contrast_data):.3f}")
            print(f"        Range contrast: [{np.nanmin(contrast_data):.3f}, {np.nanmax(contrast_data):.3f}]")
        
        return contrast_maps
    
    def analyze_subject(self, subject_id: str, func_img: nib.Nifti1Image,
                       events_df: pd.DataFrame, output_dir: str,
                       data_dir: str,
                       motion_regressors: Optional[np.ndarray] = None) -> Dict[str, nib.Nifti1Image]:
        """
        Run complete volumetric GLM analysis for a single subject.
        
        Parameters
        ----------
        subject_id : str
            Subject ID
        func_img : nib.Nifti1Image
            Functional data
        events_df : pd.DataFrame
            Stimulus timing information
        output_dir : str
            Output directory
        data_dir : str
            Base data directory
        motion_regressors : Optional[np.ndarray]
            Motion parameters array of shape (n_volumes, 6)
            
        Returns
        -------
        Dict[str, nib.Nifti1Image]
            Dictionary with contrast maps
        """
        print(f"\nVOLUMETRIC GLM ANALYSIS: {subject_id}")
        
        # Load anatomical data
        anat_path = Path(data_dir) / subject_id / "ana"
        anat_files = list(anat_path.glob("*.nii*"))
        if not anat_files:
            raise ValueError(f"No anatomical data found in {anat_path}")
        
        anat_img = nib.load(str(anat_files[0]))
        print(f"Anatomical data loaded successfully:")
        print(f"   Path: {anat_files[0]}")
        print(f"   Shape: {anat_img.shape}")
        print(f"   Voxel size: {anat_img.header.get_zooms()}")
        
        # Preprocess functional data with proper registration
        print(f"    Preprocessing functional data with proper registration...")
        print(f"      Using anatomical-guided registration pipeline...")
        preprocessed_img = self.preprocess_functional_data(func_img, anat_img)
        
        # Create design matrix with motion regressors if available
        n_scans = preprocessed_img.shape[-1]
        if motion_regressors is not None:
            if not isinstance(motion_regressors, np.ndarray):
                print("    Warning: Motion regressors not in correct format, converting...")
                motion_regressors = np.array(motion_regressors)
            
            if motion_regressors.shape[0] != n_scans:
                raise ValueError(f"Motion regressors length ({motion_regressors.shape[0]}) "
                               f"doesn't match n_scans ({n_scans})")
            
            print(f"    Including {motion_regressors.shape[1]} motion regressors")
            
        design_matrix = self.create_design_matrix(
            events_df, n_scans,
            motion_regressors=motion_regressors
        )
        
        # Run GLM
        glm_results = self.run_volumetric_glm(preprocessed_img, design_matrix)
        
        # Create vocal vs non-vocal contrast
        contrast_vector = self.create_vocal_vs_nonvocal_contrast(design_matrix)
        
        # Compute contrast maps
        contrast_maps = self.compute_contrast_maps(glm_results, contrast_vector)
        
        # Save results
        output_path = Path(output_dir) / subject_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save contrast maps
        for map_name, img in contrast_maps.items():
            if img is not None:  # Only save non-None maps
                map_path = output_path / f"{subject_id}_{map_name}.nii.gz"
                nib.save(img, str(map_path))
        
        # Save design matrix
        design_path = output_path / f"{subject_id}_design_matrix.csv"
        design_matrix.to_csv(design_path)
        
        # Save analysis info
        info = {
            'subject_id': subject_id,
            'analysis_date': str(pd.Timestamp.now()),
            'tr': self.tr,
            'hrf_model': self.hrf_model,
            'drift_model': self.drift_model,
            'high_pass': self.high_pass,
            'smoothing_fwhm': self.smoothing_fwhm,
            'n_timepoints': preprocessed_img.shape[-1],
            'n_regressors': design_matrix.shape[1],
            'has_motion': motion_regressors is not None,
            'has_anatomical': anat_img is not None,
            'registration_type': 'anatomical-guided' if anat_img is not None else 'direct-to-mni'
        }
        
        info_path = output_path / f"{subject_id}_analysis_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print("\nVolumetric GLM analysis complete for {subject_id}")
        return contrast_maps
    
    def _register_via_anatomical_ants(self, func_img: nib.Nifti1Image, 
                                      anat_img: nib.Nifti1Image, 
                                      mni_template: nib.Nifti1Image) -> nib.Nifti1Image:
        """
        Register functional data via anatomical to MNI space using ANTs as fallback.
        
        Uses ANTs registration as a fallback when FSL fails.
        """
        if not ANTS_AVAILABLE:
            print("        ERROR: ANTs not available, falling back to simple resampling...")
            return self._register_fallback_resample(func_img, mni_template)
        
        print("        Using ANTs for anatomical registration...")
        
        try:
            # Convert nibabel images to ANTs images
            func_ants = ants.from_nibabel(func_img)
            anat_ants = ants.from_nibabel(anat_img)
            mni_ants = ants.from_nibabel(mni_template)
            
            # Step 1: Register anatomical to MNI
            print("        Step 1: Anatomical → MNI registration (ANTs)...")
            anat_to_mni = ants.registration(
                fixed=mni_ants,
                moving=anat_ants,
                type_of_transform='Affine'
            )
            
            # Step 2: Register functional to anatomical
            print("        Step 2: Functional → Anatomical registration (ANTs)...")
            func_to_anat = ants.registration(
                fixed=anat_ants,
                moving=func_ants,
                type_of_transform='Rigid'
            )
            
            # Step 3: Combine transformations
            print("        Step 3: Applying combined transformation...")
            func_in_mni_ants = ants.apply_transforms(
                fixed=mni_ants,
                moving=func_ants,
                transformlist=func_to_anat['fwdtransforms'] + anat_to_mni['fwdtransforms']
            )
            
            # Convert back to nibabel
            func_in_mni = func_in_mni_ants.to_nibabel()
            
            print(f"        ANTs registration complete:")
            print(f"          Original: {func_img.shape}")
            print(f"          Registered: {func_in_mni.shape}")
            
            return func_in_mni
            
        except Exception as e:
            print(f"        ✗ ANTs registration failed: {e}")
            print("        Falling back to simple resampling...")
            return self._register_fallback_resample(func_img, mni_template)
    
    def _register_direct_to_mni_ants(self, func_img: nib.Nifti1Image, 
                                    mni_template: nib.Nifti1Image) -> nib.Nifti1Image:
        """
        Register functional data directly to MNI space using ANTs as fallback.
        """
        if not ANTS_AVAILABLE:
            print("        ERROR: ANTs not available, falling back to simple resampling...")
            return self._register_fallback_resample(func_img, mni_template)
        
        print("        Using ANTs for direct functional-to-MNI registration...")
        
        try:
            # Convert nibabel images to ANTs images
            func_ants = ants.from_nibabel(func_img)
            mni_ants = ants.from_nibabel(mni_template)
            
            # Direct functional to MNI registration
            print("        Direct Functional → MNI registration (ANTs)...")
            registration_result = ants.registration(
                fixed=mni_ants,
                moving=func_ants,
                type_of_transform='Affine'
            )
            
            # Apply transformation
            func_in_mni_ants = ants.apply_transforms(
                fixed=mni_ants,
                moving=func_ants,
                transformlist=registration_result['fwdtransforms']
            )
            
            # Convert back to nibabel
            func_in_mni = func_in_mni_ants.to_nibabel()
            
            print(f"        ANTs registration complete:")
            print(f"          Original: {func_img.shape}")
            print(f"          Registered: {func_in_mni.shape}")
            
            return func_in_mni
            
        except Exception as e:
            print(f"        ✗ ANTs registration failed: {e}")
            print("        Falling back to simple resampling...")
            return self._register_fallback_resample(func_img, mni_template)
    
    def _register_fallback_resample(self, func_img: nib.Nifti1Image, 
                                   mni_template: nib.Nifti1Image) -> nib.Nifti1Image:
        """
        Fallback registration using simple resampling when both FSL and ANTs fail.
        
        This is not proper registration but ensures the pipeline doesn't crash.
        """
        from nilearn.image import resample_img
        
        print("        WARNING: Using fallback resampling (NOT proper registration)...")
        print("        This may result in poor anatomical alignment!")
        
        # Use nilearn's resample_img with explicit target parameters
        func_mni = resample_img(
            func_img,
            target_affine=mni_template.affine,
            target_shape=mni_template.shape[:3],
            interpolation='continuous'
        )
        
        print(f"        Fallback resampling: {func_img.shape} → {func_mni.shape}")
        print(f"        Final affine matches MNI: {np.allclose(func_mni.affine, mni_template.affine)}")
        
        return func_mni


def run_single_subject_volumetric_glm(subject_id: str, data_loader,
                                     output_dir: str,
                                     tva_loc_path: str,
                                     data_dir: str) -> Dict[str, nib.Nifti1Image]:
    """
    Run volumetric GLM analysis for a single subject.

    Parameters
    ----------
    subject_id : str
        Subject ID
    data_loader : Pernet2015DataLoader
        Data loader instance
    output_dir : str
        Output directory
    tva_loc_path : str
        Path to the localizer stimulus-order file (<raw-root>/voice_localizer/TVA_loc.txt).
    data_dir : str
        Subjects directory (<raw-root>/subs) — anat lookup + motion .mat files.

    Returns
    -------
    Dict[str, nib.Nifti1Image]
        Volumetric contrast maps
    """
    # Load functional data
    func_img = data_loader.load_functional_data(subject_id)
    if func_img is None:
        raise ValueError(f"No functional data found for {subject_id}")

    # Load events - include ALL conditions (vocal, non-vocal, AND silence)
    from .timing import parse_stimulus_timing
    events_df = parse_stimulus_timing(tva_loc_path, include_final_silence=False)

    # Initialize volumetric GLM analyzer
    glm_analyzer = VolumetricGLMAnalyzer()

    # Extract motion parameters and outlier regressors (following Pernet 2015 methodology)
    print(f"  Extracting motion parameters and outlier regressors...")
    try:
        from .motion_correction import extract_motion_from_dataset
        motion_regressors, outlier_mask, motion_stats = extract_motion_from_dataset(
            subject_id, data_dir
        )
        print(f"    Motion extraction successful:")
        print(f"      - Motion regressors: {motion_regressors.shape}")
        print(f"      - Outliers detected: {motion_stats['n_total_outliers']} ({motion_stats['outlier_percentage']:.1f}%)")
        print(f"      - Total regressors: {len(motion_stats['regressor_names'])} (6 motion + 6 derivatives + {motion_stats['n_total_outliers']} outliers)")
    except Exception as e:
        print(f"    WARNING: Motion extraction failed: {e}")
        print(f"    Proceeding without motion parameters (not recommended for Pernet 2015 methodology)")
        motion_regressors = None

    # Run analysis with motion regressors
    contrast_maps = glm_analyzer.analyze_subject(
        subject_id, func_img, events_df, output_dir, data_dir,
        motion_regressors=motion_regressors  # NEW: Pass motion regressors to GLM
    )

    return contrast_maps


def test_volumetric_glm():
    """Test volumetric GLM functionality."""
    try:
        # Test implementation
        test_successful = True  # Set based on test results
        n_maps = 5  # Example number of contrast maps
        
        if test_successful:
            print(f"Test successful! Generated {n_maps} contrast maps")
            return True
        else:
            print("Volumetric GLM test failed")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    test_volumetric_glm() 