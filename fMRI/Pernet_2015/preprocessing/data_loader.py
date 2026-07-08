"""
Data loading utilities for Pernet 2015 voice localizer dataset.

This module provides functions to load anatomical and functional data
for multiple subjects from the Pernet 2015 dataset.

Port note (release repo): the baked-in dataset path was removed — `base_path`
(the raw BIDS root, e.g. `--raw-root`) is now required. Loading logic is verbatim.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings


class Pernet2015DataLoader:
    """
    Data loader for Pernet 2015 voice localizer dataset.
    
    Handles loading of anatomical and functional data for multiple subjects.
    """
    
    def __init__(self, base_path: str | None = None):
        """
        Initialize the data loader.

        Parameters
        ----------
        base_path : str
            Base path to the Pernet 2015 dataset (raw BIDS root; required).
        """
        if base_path is None:
            raise ValueError(
                "base_path is required (pass the Pernet raw BIDS root, e.g. --raw-root). "
                "There is no baked-in dataset path in the release repo."
            )
        self.base_path = Path(base_path)
        self.subs_path = self.base_path / "subs"
        self.voice_localizer_path = self.base_path / "voice_localizer"
        
        # Validate paths
        if not self.subs_path.exists():
            raise ValueError(f"Subjects directory not found: {self.subs_path}")
        if not self.voice_localizer_path.exists():
            raise ValueError(f"Voice localizer directory not found: {self.voice_localizer_path}")
    
    def get_subject_list(self) -> List[str]:
        """
        Get list of all available subjects.
        
        Returns
        -------
        List[str]
            List of subject IDs (e.g., ['sub001_Ed', 'sub002_Ed', ...])
        """
        subjects = []
        for item in self.subs_path.iterdir():
            if item.is_dir() and item.name.startswith('sub') and item.name.endswith('_Ed'):
                subjects.append(item.name)
        
        subjects.sort()  # Ensure consistent ordering
        return subjects
    
    def get_subject_paths(self, subject_id: str) -> Dict[str, Path]:
        """
        Get file paths for a specific subject.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (e.g., 'sub001_Ed')
            
        Returns
        -------
        Dict[str, Path]
            Dictionary with paths to subject's data files
        """
        subject_path = self.subs_path / subject_id
        
        if not subject_path.exists():
            raise ValueError(f"Subject directory not found: {subject_path}")
        
        # Find anatomical file
        ana_path = subject_path / "ana"
        anat_file = None
        if ana_path.exists():
            # Look for both .nii and .nii.gz files
            for pattern in ["*.nii.gz", "*.nii"]:
                for file in ana_path.glob(pattern):
                    anat_file = file
                    break
                if anat_file:
                    break
        
        # Find functional file
        func_path = subject_path / "func"
        func_file = None
        if func_path.exists():
            # Look for both .nii.gz and .nii files
            func_gz = func_path / f"{subject_id}.nii.gz"
            func_nii = func_path / f"{subject_id}.nii"
            if func_gz.exists():
                func_file = func_gz
            elif func_nii.exists():
                func_file = func_nii
        
        paths = {
            'subject_dir': subject_path,
            'anatomical': anat_file,
            'functional': func_file,
            'mat_file': func_path / f"{subject_id}.mat" if func_path.exists() else None
        }
        
        return paths
    
    def load_anatomical_data(self, subject_id: str) -> Optional[nib.Nifti1Image]:
        """
        Load anatomical data for a subject.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (e.g., 'sub001_Ed')
            
        Returns
        -------
        Optional[nib.Nifti1Image]
            Anatomical image or None if not found
        """
        paths = self.get_subject_paths(subject_id)
        
        if paths['anatomical'] is None or not paths['anatomical'].exists():
            warnings.warn(f"No anatomical data found for {subject_id}")
            return None
        
        try:
            anat_img = nib.load(str(paths['anatomical']))
            return anat_img
        except Exception as e:
            warnings.warn(f"Error loading anatomical data for {subject_id}: {e}")
            return None
    
    def load_functional_data(self, subject_id: str) -> Optional[nib.Nifti1Image]:
        """
        Load functional data for a subject.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (e.g., 'sub001_Ed')
            
        Returns
        -------
        Optional[nib.Nifti1Image]
            Functional image or None if not found
        """
        paths = self.get_subject_paths(subject_id)
        
        if paths['functional'] is None or not paths['functional'].exists():
            warnings.warn(f"No functional data found for {subject_id}")
            return None
        
        try:
            func_img = nib.load(str(paths['functional']))
            return func_img
        except Exception as e:
            warnings.warn(f"Error loading functional data for {subject_id}: {e}")
            return None
    
    def get_data_info(self, subject_id: str) -> Dict:
        """
        Get information about a subject's data.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (e.g., 'sub001_Ed')
            
        Returns
        -------
        Dict
            Dictionary with data information
        """
        paths = self.get_subject_paths(subject_id)
        
        info = {
            'subject_id': subject_id,
            'has_anatomical': paths['anatomical'] is not None and paths['anatomical'].exists(),
            'has_functional': paths['functional'] is not None and paths['functional'].exists(),
            'has_mat_file': paths['mat_file'] is not None and paths['mat_file'].exists(),
            'paths': paths
        }
        
        # Add anatomical info if available
        if info['has_anatomical']:
            anat_img = self.load_anatomical_data(subject_id)
            if anat_img is not None:
                info['anatomical_shape'] = anat_img.shape
                info['anatomical_affine'] = anat_img.affine
        
        # Add functional info if available
        if info['has_functional']:
            func_img = self.load_functional_data(subject_id)
            if func_img is not None:
                info['functional_shape'] = func_img.shape
                info['functional_affine'] = func_img.affine
                info['n_trs'] = func_img.shape[-1]
                info['estimated_duration'] = func_img.shape[-1] * 2.0  # Assuming TR=2s
        
        return info
    
    def load_multiple_subjects(self, subject_ids: List[str], 
                             data_type: str = 'functional') -> Dict[str, Union[nib.Nifti1Image, None]]:
        """
        Load data for multiple subjects.
        
        Parameters
        ----------
        subject_ids : List[str]
            List of subject IDs to load
        data_type : str
            Type of data to load ('functional' or 'anatomical')
            
        Returns
        -------
        Dict[str, Union[nib.Nifti1Image, None]]
            Dictionary mapping subject IDs to their data
        """
        data = {}
        
        for subject_id in subject_ids:
            if data_type == 'functional':
                data[subject_id] = self.load_functional_data(subject_id)
            elif data_type == 'anatomical':
                data[subject_id] = self.load_anatomical_data(subject_id)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
        
        return data
    
    def validate_subject_data(self, subject_id: str) -> Dict[str, bool]:
        """
        Validate that a subject has all required data files.
        
        Parameters
        ----------
        subject_id : str
            Subject ID to validate
            
        Returns
        -------
        Dict[str, bool]
            Dictionary with validation results
        """
        paths = self.get_subject_paths(subject_id)
        
        validation = {
            'subject_exists': True,
            'anatomical_exists': paths['anatomical'] is not None and paths['anatomical'].exists(),
            'functional_exists': paths['functional'] is not None and paths['functional'].exists(),
            'mat_file_exists': paths['mat_file'] is not None and paths['mat_file'].exists()
        }
        
        # Check if files can be loaded
        if validation['anatomical_exists']:
            anat_img = self.load_anatomical_data(subject_id)
            validation['anatomical_loadable'] = anat_img is not None
        
        if validation['functional_exists']:
            func_img = self.load_functional_data(subject_id)
            validation['functional_loadable'] = func_img is not None
            if func_img is not None:
                validation['functional_has_310_trs'] = func_img.shape[-1] == 310
        
        return validation

    def load_motion_parameters(self, subject_id: str) -> Optional[np.ndarray]:
        """
        Load motion parameters from subject's .mat file.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (e.g., 'sub001_Ed')
            
        Returns
        -------
        Optional[np.ndarray]
            Motion parameters array of shape (n_volumes, 6) or None if not found
        """
        paths = self.get_subject_paths(subject_id)
        
        if paths['mat_file'] is None or not paths['mat_file'].exists():
            warnings.warn(f"No motion parameters (.mat file) found for {subject_id}")
            return None
        
        try:
            from scipy.io import loadmat
            mat_data = loadmat(str(paths['mat_file']))
            
            # Extract motion parameters (should be 6 parameters per volume)
            motion_params = mat_data.get('rp_I', None)
            
            if motion_params is None:
                warnings.warn(f"No motion parameters found in .mat file for {subject_id}")
                return None
                
            # Ensure correct shape (n_volumes, 6)
            if motion_params.shape[1] != 6:
                motion_params = motion_params.T
                
            if motion_params.shape[1] != 6:
                warnings.warn(f"Unexpected motion parameter shape for {subject_id}: {motion_params.shape}")
                return None
                
            return motion_params.astype(np.float32)
            
        except Exception as e:
            warnings.warn(f"Error loading motion parameters for {subject_id}: {e}")
            return None


def load_subject_data(subject_id: str, base_path: str | None = None) -> Dict:
    """
    Convenience function to load all data for a single subject.
    
    Parameters
    ----------
    subject_id : str
        Subject ID (e.g., 'sub001_Ed')
    base_path : str
        Base path to the dataset
        
    Returns
    -------
    Dict
        Dictionary with subject's data and information
    """
    loader = Pernet2015DataLoader(base_path)
    
    data = {
        'subject_id': subject_id,
        'anatomical': loader.load_anatomical_data(subject_id),
        'functional': loader.load_functional_data(subject_id),
        'info': loader.get_data_info(subject_id),
        'validation': loader.validate_subject_data(subject_id)
    }
    
    return data


def get_dataset_summary(base_path: str | None = None) -> Dict:
    """
    Get a summary of the entire dataset.
    
    Parameters
    ----------
    base_path : str
        Base path to the dataset
        
    Returns
    -------
    Dict
        Dictionary with dataset summary information
    """
    loader = Pernet2015DataLoader(base_path)
    subjects = loader.get_subject_list()
    
    summary = {
        'total_subjects': len(subjects),
        'subject_list': subjects,
        'data_availability': {
            'anatomical': 0,
            'functional': 0,
            'mat_files': 0
        },
        'functional_trs': [],
        'validation_results': {}
    }
    
    # Check data availability for each subject
    for subject_id in subjects:
        validation = loader.validate_subject_data(subject_id)
        
        if validation['anatomical_exists']:
            summary['data_availability']['anatomical'] += 1
        if validation['functional_exists']:
            summary['data_availability']['functional'] += 1
        if validation['mat_file_exists']:
            summary['data_availability']['mat_files'] += 1
        
        # Get TR information
        if validation['functional_exists']:
            func_img = loader.load_functional_data(subject_id)
            if func_img is not None:
                summary['functional_trs'].append(func_img.shape[-1])
        
        summary['validation_results'][subject_id] = validation
    
    # Add statistics
    if summary['functional_trs']:
        summary['tr_statistics'] = {
            'mean': np.mean(summary['functional_trs']),
            'std': np.std(summary['functional_trs']),
            'min': np.min(summary['functional_trs']),
            'max': np.max(summary['functional_trs']),
            'expected': 310
        }
    
    return summary


if __name__ == "__main__":
    import sys
    print("Pernet2015DataLoader requires an explicit base_path (raw BIDS root).")
    print("Usage: Pernet2015DataLoader(base_path='/path/to/pernet_2015')")
    sys.exit(0)