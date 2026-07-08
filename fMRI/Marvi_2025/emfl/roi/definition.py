#!/usr/bin/env python3
"""
Define Functional Regions of Interest (fROIs) using Anatomical Parcels

This script implements the fROI definition methodology from Marvi et al. (2025):
1. Load anatomical parcels (from template space)
2. Resample parcels to subject's functional space
3. Load GLM contrast maps (z-scores)
4. Select top N% most significant voxels within each parcel
5. Save fROI masks for each subject and contrast

This implements the "anatomical constraint + functional selection" approach
described in the original paper.

Usage:
------
python 13_define_frois.py --subject sub-kaneff01 --contrast faces_vs_objects --percentile 10
python 13_define_frois.py --subject sub-kaneff01 --run-split odd --percentile 10
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import nibabel as nib
from nilearn import image
import pandas as pd

# Import parcel-to-contrast mapping from config
from emfl.config import PARCEL_CONTRAST_MAP


class fROIDefiner:
    """
    Define functional ROIs using anatomical constraint + functional selection.
    
    Parameters
    ----------
    parcels_dir : Path
        Directory containing anatomical parcel files
    derivatives_dir : Path
        fMRIprep derivatives directory with GLM outputs
    subject_id : str
        Subject identifier
    space : str
        Analysis space ('MNI152NLin2009cAsym' or 'T1w')
    """
    
    def __init__(self, parcels_dir: Path, derivatives_dir: Path, 
                 subject_id: str, space: str = 'MNI152NLin2009cAsym'):
        self.parcels_dir = Path(parcels_dir)
        self.derivatives_dir = Path(derivatives_dir)
        self.subject_id = subject_id
        self.space = space
        
        self.subject_dir = self.derivatives_dir / subject_id
        self.glm_dir = self.subject_dir / 'first_level_glm'
        
        if not self.glm_dir.exists():
            raise FileNotFoundError(f"GLM directory not found: {self.glm_dir}")
    
    def warp_mni_parcel_to_t1w(self, parcel_img: nib.Nifti1Image, 
                              parcel_name: str, hemisphere: str = None) -> nib.Nifti1Image:
        """
        Warp MNI parcel to subject's T1w space using fMRIprep transforms.
        
        Uses nipype interface to ANTs for applying the transformation.
        Results are cached to avoid repeated warping.
        
        Parameters
        ----------
        parcel_img : nib.Nifti1Image
            Parcel in MNI space
        parcel_name : str
            Name of parcel (for caching)
        hemisphere : str, optional
            Hemisphere ('lh' or 'rh')
            
        Returns
        -------
        nib.Nifti1Image
            Warped parcel in T1w space
        """
        import tempfile
        import os
        import subprocess
        
        # Build cache path
        hemi_str = f"_{hemisphere}" if hemisphere else ""
        cache_dir = self.subject_dir / 'parcels_t1w'
        cache_dir.mkdir(exist_ok=True)
        cached_parcel = cache_dir / f"{parcel_name}{hemi_str}_space-T1w.nii.gz"
        
        # Return cached if exists
        if cached_parcel.exists():
            return nib.load(cached_parcel)
        
        # Find transform file
        transform_file = self.subject_dir / 'anat' / f'{self.subject_id}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'
        if not transform_file.exists():
            raise FileNotFoundError(f"MNI→T1w transform not found: {transform_file}")
        
        # Find reference image: use T1w BOLD data (functional resolution, unmasked)
        # This is what fMRIprep actually registered, so transforms will work correctly
        func_dir = self.subject_dir / 'func'
        bold_files = list(func_dir.glob(f'{self.subject_id}_task-effloc_run-*_space-T1w_desc-preproc_bold.nii.gz'))
        
        if not bold_files:
            raise FileNotFoundError(f"No T1w BOLD files found in {func_dir}. Cannot warp parcels.")
        
        # Extract first volume from 4D BOLD to use as 3D reference
        bold_4d = nib.load(bold_files[0])
        bold_3d = nib.Nifti1Image(bold_4d.get_fdata()[..., 0], bold_4d.affine, bold_4d.header)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='_ref.nii.gz', delete=False) as tmp_ref:
            nib.save(bold_3d, tmp_ref.name)
            reference_file = tmp_ref.name
        
        # Save parcel to temp file
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_input:
            nib.save(parcel_img, tmp_input.name)
            tmp_input_path = tmp_input.name
        
        try:
            # Use ANTs via fMRIprep container (Apptainer)
            container_path = '/work/upschrimpf1/mehrer/fmriprep-24.0.1.simg'
            
            cmd = [
                'apptainer', 'exec',
                '--bind', f'{tmp_input_path}:{tmp_input_path}:ro',
                '--bind', f'{str(reference_file)}:{str(reference_file)}:ro',
                '--bind', f'{str(transform_file)}:{str(transform_file)}:ro',
                '--bind', f'{str(cached_parcel.parent)}:{str(cached_parcel.parent)}',
                container_path,
                'antsApplyTransforms',
                '-d', '3',
                '-i', tmp_input_path,
                '-r', str(reference_file),  # Use functional z-map as reference (2mm)
                '-t', str(transform_file),
                '-o', str(cached_parcel),
                '-n', 'GenericLabel'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"antsApplyTransforms failed: {result.stderr}")
            
            # Load and return warped parcel
            warped_parcel = nib.load(cached_parcel)
            
            return warped_parcel
            
        finally:
            # Cleanup temp files
            try:
                os.unlink(tmp_input_path)
            except:
                pass
            try:
                os.unlink(reference_file)
            except:
                pass
    
    def load_parcel(self, parcel_category: str, parcel_name: str, 
                   hemisphere: str = None) -> nib.Nifti1Image:
        """
        Load an anatomical parcel mask.
        
        Parameters
        ----------
        parcel_category : str
            Category of parcel (e.g., 'julian', 'language', 'tom')
        parcel_name : str
            Name of the parcel (e.g., 'ffa', 'ppa')
        hemisphere : str, optional
            'lh' or 'rh' for hemisphere-specific parcels
            
        Returns
        -------
        nib.Nifti1Image
            Parcel mask image
        """
        if hemisphere:
            parcel_file = self.parcels_dir / parcel_category / f"{hemisphere}.{parcel_name}.nii.gz"
        else:
            parcel_file = self.parcels_dir / parcel_category / f"{parcel_name}.nii.gz"
        
        if not parcel_file.exists():
            raise FileNotFoundError(f"Parcel not found: {parcel_file}")
        
        parcel_img = nib.load(parcel_file)
        
        # Handle 4D parcels with trailing dimension (e.g., language parcels with shape (x,y,z,1))
        if parcel_img.ndim == 4 and parcel_img.shape[3] == 1:
            parcel_data = np.squeeze(parcel_img.get_fdata())
            parcel_img = nib.Nifti1Image(parcel_data, parcel_img.affine, parcel_img.header)
        
        # MNI parcels used directly in MNI space (no warping needed)
        # If T1w analysis is needed, warping would be required here
        return parcel_img
    
    def resample_parcel_to_functional(self, parcel_img: nib.Nifti1Image, 
                                     target_img: nib.Nifti1Image) -> nib.Nifti1Image:
        """
        Resample parcel from template space to functional space.
        
        Parameters
        ----------
        parcel_img : nib.Nifti1Image
            Parcel in template space
        target_img : nib.Nifti1Image
            Target functional image (defines output space)
            
        Returns
        -------
        nib.Nifti1Image
            Resampled parcel in functional space
        """
        # Resample parcel to match functional image
        # Use nearest neighbor interpolation to preserve binary mask
        resampled = image.resample_to_img(
            parcel_img, 
            target_img, 
            interpolation='nearest'
        )
        return resampled
    
    def load_contrast_maps(self, contrast_name: str, modality: str,
                          run_split: str = 'all') -> List[nib.Nifti1Image]:
        """
        Load contrast z-score maps for all runs.
        
        Parameters
        ----------
        contrast_name : str
            Name of the contrast
        modality : str
            'visual' or 'auditory'
        run_split : str
            'all', 'even', or 'odd'
            
        Returns
        -------
        List[nib.Nifti1Image]
            List of contrast maps (one per run)
        """
        # Determine directory based on run_split
        if run_split == 'all':
            base_dir = self.glm_dir / f'effloc_{modality}'
        else:
            base_dir = self.glm_dir / f'effloc_{modality}_split-{run_split}'
        
        if not base_dir.exists():
            raise FileNotFoundError(f"GLM directory not found: {base_dir}")
        
        # Determine which runs to load
        if run_split == 'even':
            runs = ['002', '004']
        elif run_split == 'odd':
            runs = ['001', '003', '005']
        else:  # 'all'
            runs = ['001', '002', '003', '004', '005']
        
        contrast_maps = []
        for run in runs:
            # Build filename based on space (MNI includes res-2, T1w doesn't)
            if 'MNI' in self.space:
                contrast_file = (base_dir / f'run-{run}' / 
                               f'{self.subject_id}_task-effloc_run-{run}_{modality}_{contrast_name}_space-{self.space}_res-2_zmap.nii.gz')
            else:
                contrast_file = (base_dir / f'run-{run}' / 
                               f'{self.subject_id}_task-effloc_run-{run}_{modality}_{contrast_name}_space-{self.space}_zmap.nii.gz')
            
            if contrast_file.exists():
                contrast_maps.append(nib.load(contrast_file))
            else:
                print(f"  Warning: Contrast file not found: {contrast_file.name}")
        
        if len(contrast_maps) == 0:
            raise FileNotFoundError(f"No contrast maps found for {contrast_name} in {base_dir}")
        
        return contrast_maps
    
    def select_top_voxels(self, parcel_data: np.ndarray, 
                         contrast_data: np.ndarray, 
                         percentile: float = 10) -> np.ndarray:
        """
        Select top N% most significant voxels within parcel.
        
        Parameters
        ----------
        parcel_data : np.ndarray
            Binary parcel mask
        contrast_data : np.ndarray
            Z-score contrast map
        percentile : float
            Percentage of voxels to select (default: 10%)
            
        Returns
        -------
        np.ndarray
            Binary mask of selected voxels
        """
        # Get voxels within parcel
        parcel_mask = parcel_data > 0
        
        # Extract z-scores within parcel
        z_scores_in_parcel = contrast_data[parcel_mask]
        
        # Calculate threshold for top N%
        if len(z_scores_in_parcel) == 0:
            print("  Warning: No voxels found in parcel")
            return np.zeros_like(parcel_data, dtype=bool)
        
        threshold = np.percentile(z_scores_in_parcel, 100 - percentile)
        
        # Create fROI mask
        froi_mask = np.zeros_like(parcel_data, dtype=bool)
        froi_mask[parcel_mask] = contrast_data[parcel_mask] >= threshold
        
        return froi_mask
    
    def define_froi(self, parcel_category: str, parcel_name: str,
                   hemisphere: str = None, percentile: float = 10,
                   run_split: str = 'all') -> Tuple[nib.Nifti1Image, Dict]:
        """
        Define fROI using anatomical parcel and functional contrast.
        
        Parameters
        ----------
        parcel_category : str
            Category of parcel
        parcel_name : str
            Name of the parcel
        hemisphere : str, optional
            Hemisphere ('lh' or 'rh')
        percentile : float
            Top N% of voxels to select
        run_split : str
            Which runs to use ('all', 'even', 'odd')
            
        Returns
        -------
        nib.Nifti1Image
            fROI mask image
        Dict
            Metadata about the fROI
        """
        print(f"\nDefining fROI: {parcel_name} ({hemisphere if hemisphere else 'bilateral'})")
        print(f"  Parcel category: {parcel_category}")
        print(f"  Run split: {run_split}")
        print(f"  Percentile: top {percentile}%")
        
        # Determine contrast and modality
        contrast_name = PARCEL_CONTRAST_MAP.get(parcel_name)
        if contrast_name is None:
            raise ValueError(f"No contrast mapping found for parcel: {parcel_name}")
        
        # Determine modality from contrast name
        visual_contrasts = ['faces_vs_objects', 'scenes_vs_objects', 'bodies_vs_objects', 
                          'words_vs_objects', 'objects_vs_words']
        modality = 'visual' if contrast_name in visual_contrasts else 'auditory'
        
        print(f"  Using contrast: {contrast_name} ({modality})")
        
        # Load parcel
        parcel_img = self.load_parcel(parcel_category, parcel_name, hemisphere)
        print(f"  Parcel loaded: {np.sum(parcel_img.get_fdata() > 0)} voxels")
        
        # Load contrast maps
        contrast_maps = self.load_contrast_maps(contrast_name, modality, run_split)
        print(f"  Loaded {len(contrast_maps)} contrast maps")
        
        # Average contrast maps across runs
        contrast_data_list = [img.get_fdata() for img in contrast_maps]
        mean_contrast = np.mean(contrast_data_list, axis=0)
        print(f"  Mean contrast calculated across {len(contrast_maps)} runs")
        
        # Resample parcel to match functional space
        parcel_resampled = self.resample_parcel_to_functional(parcel_img, contrast_maps[0])
        parcel_data = parcel_resampled.get_fdata()
        print(f"  Parcel resampled: {np.sum(parcel_data > 0)} voxels in functional space")
        
        # Select top voxels
        froi_mask = self.select_top_voxels(parcel_data, mean_contrast, percentile)
        n_voxels = np.sum(froi_mask)
        print(f"  fROI defined: {n_voxels} voxels (top {percentile}%)")
        
        # Create fROI image
        froi_img = nib.Nifti1Image(froi_mask.astype(np.float32), 
                                   contrast_maps[0].affine,
                                   contrast_maps[0].header)
        
        # Metadata
        metadata = {
            'parcel_name': parcel_name,
            'parcel_category': parcel_category,
            'hemisphere': hemisphere,
            'contrast': contrast_name,
            'modality': modality,
            'run_split': run_split,
            'percentile': percentile,
            'n_voxels': int(n_voxels),
            'n_runs': len(contrast_maps),
            'parcel_voxels': int(np.sum(parcel_data > 0))
        }
        
        return froi_img, metadata
    
    def save_froi(self, froi_img: nib.Nifti1Image, metadata: Dict, 
                 output_dir: Path = None):
        """
        Save fROI mask and metadata.
        
        Parameters
        ----------
        froi_img : nib.Nifti1Image
            fROI mask image
        metadata : Dict
            fROI metadata
        output_dir : Path, optional
            Output directory (default: derivatives/sub-*/frois/)
        """
        if output_dir is None:
            output_dir = self.subject_dir / 'frois' / f'space-{self.space}'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        parcel_name = metadata['parcel_name']
        hemisphere = metadata['hemisphere']
        run_split = metadata['run_split']
        
        if hemisphere:
            roi_name = f"{hemisphere}_{parcel_name}"
        else:
            roi_name = parcel_name
        
        if run_split != 'all':
            filename = f"{self.subject_id}_roi-{roi_name}_split-{run_split}_space-{self.space}_mask.nii.gz"
        else:
            filename = f"{self.subject_id}_roi-{roi_name}_space-{self.space}_mask.nii.gz"
        
        output_path = output_dir / filename
        nib.save(froi_img, output_path)
        print(f"  Saved: {output_path.name}")
        
        # Save metadata as JSON
        import json
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata: {metadata_path.name}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Define fROIs using anatomical parcels and functional contrasts',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--subject', type=str, required=True,
                       help='Subject ID (e.g., sub-kaneff01)')
    parser.add_argument('--parcel-category', type=str, default='julian',
                       choices=['julian', 'language', 'tom', 'md', 'speech', 'vwfa'],
                       help='Parcel category (default: julian)')
    parser.add_argument('--parcel-name', type=str, required=True,
                       help='Parcel name (e.g., ffa, ppa, eba)')
    parser.add_argument('--hemisphere', type=str, choices=['lh', 'rh'],
                       help='Hemisphere (if applicable)')
    parser.add_argument('--percentile', type=float, default=10,
                       help='Top N%% of voxels to select (default: 10)')
    parser.add_argument('--run-split', type=str, default='all',
                       choices=['all', 'even', 'odd'],
                       help='Which runs to use (default: all)')
    parser.add_argument('--space', type=str, default='MNI152NLin2009cAsym',
                       help='Analysis space (default: MNI152NLin2009cAsym)')
    parser.add_argument('--parcels-dir', type=str,
                       default='src/aux/emfl_analysis-main/PARCELS',
                       help='Directory containing parcels')
    parser.add_argument('--derivatives-dir', type=str,
                       default='/work/upschrimpf1/mehrer/datasets/Marvi_2025_efficient_fMRI_localizer/derivatives',
                       help='fMRIprep derivatives directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("fROI Definition Tool")
    print("="*70)
    print(f"Subject: {args.subject}")
    print(f"Parcel: {args.parcel_name} ({args.parcel_category})")
    if args.hemisphere:
        print(f"Hemisphere: {args.hemisphere}")
    print(f"Percentile: top {args.percentile}%")
    print(f"Run split: {args.run_split}")
    print(f"Space: {args.space}")
    print("="*70)
    
    # Initialize definer
    definer = fROIDefiner(
        parcels_dir=args.parcels_dir,
        derivatives_dir=args.derivatives_dir,
        subject_id=args.subject,
        space=args.space
    )
    
    # Define fROI
    froi_img, metadata = definer.define_froi(
        parcel_category=args.parcel_category,
        parcel_name=args.parcel_name,
        hemisphere=args.hemisphere,
        percentile=args.percentile,
        run_split=args.run_split
    )
    
    # Save fROI
    output_path = definer.save_froi(froi_img, metadata)
    
    print("\n" + "="*70)
    print("fROI Definition Complete!")
    print("="*70)
    print(f"ROI: {metadata['parcel_name']} ({metadata['hemisphere'] if metadata['hemisphere'] else 'bilateral'})")
    print(f"Voxels: {metadata['n_voxels']}")
    print(f"Output: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()

