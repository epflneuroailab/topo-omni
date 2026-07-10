#!/usr/bin/env python3
"""
Extract ROI Responses from Defined fROIs

This script extracts beta weights or contrast values from defined fROIs.
It supports cross-validation by extracting from independent data splits.

Usage:
------
# Extract betas from fROI defined on all runs
python 14_extract_roi_responses.py --subject sub-kaneff01 --roi-mask path/to/mask.nii.gz

# Cross-validation: extract from even runs using fROI defined on odd runs
python 14_extract_roi_responses.py --subject sub-kaneff01 --roi-mask odd_mask.nii.gz --extract-from even
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import nibabel as nib
import pandas as pd
import json


class ROIResponseExtractor:
    """
    Extract responses (beta weights or contrasts) from defined fROIs.
    
    Parameters
    ----------
    derivatives_dir : Path
        fMRIprep derivatives directory
    subject_id : str
        Subject identifier
    space : str
        Analysis space
    """
    
    def __init__(self, derivatives_dir: Path, subject_id: str, space: str = 'MNI152NLin2009cAsym'):
        self.derivatives_dir = Path(derivatives_dir)
        self.subject_id = subject_id
        self.space = space
        
        self.subject_dir = self.derivatives_dir / subject_id
        self.glm_dir = self.subject_dir / 'first_level_glm'
    
    def load_roi_mask(self, roi_mask_path: Path) -> Tuple[nib.Nifti1Image, Dict]:
        """
        Load fROI mask and its metadata.
        
        Parameters
        ----------
        roi_mask_path : Path
            Path to fROI mask file
            
        Returns
        -------
        nib.Nifti1Image
            ROI mask image
        Dict
            ROI metadata
        """
        mask_img = nib.load(roi_mask_path)
        
        # Load metadata if available
        metadata_path = roi_mask_path.with_suffix('.json')
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return mask_img, metadata
    
    def extract_mean_response(self, contrast_img: nib.Nifti1Image, 
                             mask_img: nib.Nifti1Image) -> float:
        """
        Extract mean response from ROI.
        
        Parameters
        ----------
        contrast_img : nib.Nifti1Image
            Contrast z-score map
        mask_img : nib.Nifti1Image
            ROI mask
            
        Returns
        -------
        float
            Mean z-score within ROI
        """
        contrast_data = contrast_img.get_fdata()
        mask_data = mask_img.get_fdata()
        
        roi_voxels = mask_data > 0
        if np.sum(roi_voxels) == 0:
            return np.nan
        
        mean_response = np.mean(contrast_data[roi_voxels])
        return float(mean_response)
    
    def extract_roi_responses(self, roi_mask_path, 
                             contrasts: List[str] = None,
                             run_split: str = 'all') -> pd.DataFrame:
        """
        Extract responses for all contrasts from an fROI.
        
        Parameters
        ----------
        roi_mask_path : Path or str
            Path to fROI mask
        contrasts : List[str], optional
            List of contrast names to extract (default: all available)
        run_split : str
            Which runs to extract from ('all', 'even', 'odd')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: run, modality, contrast, response
        """
        # Convert to Path if string
        roi_mask_path = Path(roi_mask_path) if isinstance(roi_mask_path, str) else roi_mask_path
        
        print(f"\nExtracting responses from: {roi_mask_path.name}")
        
        # Load ROI mask
        mask_img, metadata = self.load_roi_mask(roi_mask_path)
        n_voxels = int(np.sum(mask_img.get_fdata() > 0))
        print(f"  ROI: {metadata.get('parcel_name', 'unknown')}")
        print(f"  Voxels: {n_voxels}")
        print(f"  Extract from: {run_split} runs")
        
        # Determine which runs to extract
        if run_split == 'even':
            runs = ['002', '004']
        elif run_split == 'odd':
            runs = ['001', '003', '005']
        else:  # 'all'
            runs = ['001', '002', '003', '004', '005']
        
        # Determine modalities based on contrasts or metadata
        modalities = []
        if contrasts:
            visual_contrasts = ['faces_vs_objects', 'scenes_vs_objects', 'bodies_vs_objects',
                              'words_vs_objects', 'objects_vs_words']
            if any(c in visual_contrasts for c in contrasts):
                modalities.append('visual')
            if any(c not in visual_contrasts for c in contrasts):
                modalities.append('auditory')
        else:
            # Use metadata to determine modality, or extract both
            if 'modality' in metadata:
                modalities = [metadata['modality']]
            else:
                modalities = ['visual', 'auditory']
        
        # Extract responses
        results = []
        
        for modality in modalities:
            # Determine directory
            if run_split == 'all':
                base_dir = self.glm_dir / f'effloc_{modality}'
            else:
                base_dir = self.glm_dir / f'effloc_{modality}_split-{run_split}'
            
            if not base_dir.exists():
                print(f"  Warning: Directory not found: {base_dir}")
                continue
            
            # Get all contrast files if contrasts not specified
            if contrasts is None:
                # Get from first run
                first_run_dir = base_dir / f'run-{runs[0]}'
                if first_run_dir.exists():
                    if 'MNI' in self.space:
                        pattern = f'{self.subject_id}_task-effloc_run-{runs[0]}_{modality}_*_space-{self.space}_res-2_zmap.nii.gz'
                    else:
                        pattern = f'{self.subject_id}_task-effloc_run-{runs[0]}_{modality}_*_space-{self.space}_zmap.nii.gz'
                    
                    contrast_files = list(first_run_dir.glob(pattern))
                    contrasts = []
                    for f in contrast_files:
                        # Extract contrast name from filename
                        parts = f.stem.split('_')
                        # Find indices
                        try:
                            mod_idx = parts.index(modality)
                            space_idx = [i for i, p in enumerate(parts) if 'space-' in p][0]
                            contrast_name = '_'.join(parts[mod_idx+1:space_idx])
                            contrasts.append(contrast_name)
                        except:
                            continue
            
            print(f"  Modality: {modality} ({len(contrasts)} contrasts)")
            
            for contrast_name in contrasts:
                for run in runs:
                    # Build filename
                    if 'MNI' in self.space:
                        contrast_file = (base_dir / f'run-{run}' /
                                       f'{self.subject_id}_task-effloc_run-{run}_{modality}_{contrast_name}_space-{self.space}_res-2_zmap.nii.gz')
                    else:
                        contrast_file = (base_dir / f'run-{run}' /
                                       f'{self.subject_id}_task-effloc_run-{run}_{modality}_{contrast_name}_space-{self.space}_zmap.nii.gz')
                    
                    if not contrast_file.exists():
                        continue
                    
                    # Load and extract
                    contrast_img = nib.load(contrast_file)
                    mean_response = self.extract_mean_response(contrast_img, mask_img)
                    
                    results.append({
                        'subject': self.subject_id,
                        'roi': metadata.get('parcel_name', roi_mask_path.stem),
                        'hemisphere': metadata.get('hemisphere', 'bilateral'),
                        'run': run,
                        'modality': modality,
                        'contrast': contrast_name,
                        'response': mean_response,
                        'n_voxels': n_voxels,
                        'run_split': run_split
                    })
        
        df = pd.DataFrame(results)
        print(f"  Extracted: {len(df)} responses ({len(df['contrast'].unique())} contrasts × {len(df['run'].unique())} runs)")
        
        return df
    
    def extract_condition_responses(self, roi_mask_path, 
                                   conditions: List[str] = None,
                                   run_split: str = 'all') -> pd.DataFrame:
        """
        Extract per-condition beta estimates (effect sizes) from an fROI.
        
        This extracts individual condition responses (faces, objects, scenes, etc.)
        rather than contrasts, matching the paper's Figure 4 methodology.
        
        Parameters
        ----------
        roi_mask_path : Path or str
            Path to fROI mask
        conditions : List[str], optional
            List of condition names to extract (default: all available)
        run_split : str
            Which runs to extract from ('all', 'even', 'odd')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: run, modality, condition, beta, n_voxels, run_split
        """
        # Convert to Path if string
        roi_mask_path = Path(roi_mask_path) if isinstance(roi_mask_path, str) else roi_mask_path
        
        print(f"\nExtracting condition betas from: {roi_mask_path.name}")
        
        # Load ROI mask
        mask_img, metadata = self.load_roi_mask(roi_mask_path)
        n_voxels = int(np.sum(mask_img.get_fdata() > 0))
        print(f"  ROI: {metadata.get('parcel_name', 'unknown')}")
        print(f"  Voxels: {n_voxels}")
        print(f"  Extract from: {run_split} runs")
        
        # Determine which runs to extract
        if run_split == 'even':
            runs = ['002', '004']
        elif run_split == 'odd':
            runs = ['001', '003', '005']
        else:  # 'all'
            runs = ['001', '002', '003', '004', '005']
        
        # Extract responses
        results = []
        
        for modality in ['visual', 'auditory']:
            # Determine directory
            if run_split == 'all':
                base_dir = self.glm_dir / f'effloc_{modality}'
            else:
                base_dir = self.glm_dir / f'effloc_{modality}_split-{run_split}'
            
            if not base_dir.exists():
                print(f"  Warning: Directory not found: {base_dir}")
                continue
            
            # Get all condition effect files if conditions not specified
            if conditions is None:
                # Get from first run
                first_run_dir = base_dir / f'run-{runs[0]}'
                if first_run_dir.exists():
                    if 'MNI' in self.space:
                        pattern = f'{self.subject_id}_task-effloc_run-{runs[0]}_{modality}_*_space-{self.space}_res-2_effect.nii.gz'
                    else:
                        pattern = f'{self.subject_id}_task-effloc_run-{runs[0]}_{modality}_*_space-{self.space}_effect.nii.gz'
                    
                    effect_files = list(first_run_dir.glob(pattern))
                    conditions_to_extract = []
                    for f in effect_files:
                        # Extract condition name from filename
                        parts = f.stem.split('_')
                        # Find indices
                        try:
                            mod_idx = parts.index(modality)
                            space_idx = [i for i, p in enumerate(parts) if 'space-' in p][0]
                            condition_name = '_'.join(parts[mod_idx+1:space_idx])
                            conditions_to_extract.append(condition_name)
                        except:
                            continue
                else:
                    conditions_to_extract = []
            else:
                conditions_to_extract = conditions
            
            if not conditions_to_extract:
                continue
                
            print(f"  Modality: {modality} ({len(conditions_to_extract)} conditions)")
            
            for condition_name in conditions_to_extract:
                for run in runs:
                    # Build filename for effect map
                    if 'MNI' in self.space:
                        effect_file = (base_dir / f'run-{run}' /
                                      f'{self.subject_id}_task-effloc_run-{run}_{modality}_{condition_name}_space-{self.space}_res-2_effect.nii.gz')
                    else:
                        effect_file = (base_dir / f'run-{run}' /
                                      f'{self.subject_id}_task-effloc_run-{run}_{modality}_{condition_name}_space-{self.space}_effect.nii.gz')
                    
                    if not effect_file.exists():
                        continue
                    
                    # Load and extract mean beta (effect size)
                    effect_img = nib.load(effect_file)
                    mean_beta = self.extract_mean_response(effect_img, mask_img)
                    
                    results.append({
                        'subject': self.subject_id,
                        'roi': metadata.get('parcel_name', roi_mask_path.stem),
                        'hemisphere': metadata.get('hemisphere', 'bilateral'),
                        'run': run,
                        'modality': modality,
                        'condition': condition_name,
                        'beta': mean_beta,
                        'n_voxels': n_voxels,
                        'run_split': run_split
                    })
        
        df = pd.DataFrame(results)
        print(f"  Extracted: {len(df)} responses ({len(df['condition'].unique())} conditions × {len(df['run'].unique())} runs)")
        
        return df
    
    def compute_selectivity(self, responses_df: pd.DataFrame, 
                           preferred_contrast: str,
                           non_preferred_contrasts: List[str]) -> Dict:
        """
        Compute selectivity: preferred vs non-preferred responses.
        
        Parameters
        ----------
        responses_df : pd.DataFrame
            DataFrame with responses
        preferred_contrast : str
            Preferred contrast for this ROI
        non_preferred_contrasts : List[str]
            Non-preferred contrasts
            
        Returns
        -------
        Dict
            Selectivity metrics
        """
        pref_responses = responses_df[responses_df['contrast'] == preferred_contrast]['response'].values
        
        non_pref_responses = []
        for contrast in non_preferred_contrasts:
            vals = responses_df[responses_df['contrast'] == contrast]['response'].values
            non_pref_responses.extend(vals)
        
        if len(pref_responses) == 0 or len(non_pref_responses) == 0:
            return {
                'preferred_mean': np.nan,
                'non_preferred_mean': np.nan,
                'selectivity': np.nan,
                'effect_size': np.nan
            }
        
        pref_mean = np.mean(pref_responses)
        non_pref_mean = np.mean(non_pref_responses)
        selectivity = pref_mean - non_pref_mean
        
        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(pref_responses) + np.var(non_pref_responses)) / 2)
        effect_size = selectivity / pooled_std if pooled_std > 0 else np.nan
        
        return {
            'preferred_mean': float(pref_mean),
            'non_preferred_mean': float(non_pref_mean),
            'selectivity': float(selectivity),
            'effect_size': float(effect_size),
            'n_preferred': len(pref_responses),
            'n_non_preferred': len(non_pref_responses)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Extract responses from defined fROIs'
    )
    
    parser.add_argument('--subject', type=str, required=True,
                       help='Subject ID')
    parser.add_argument('--roi-mask', type=str, required=True,
                       help='Path to ROI mask file')
    parser.add_argument('--contrasts', nargs='+',
                       help='Specific contrasts to extract (default: all)')
    parser.add_argument('--extract-from', type=str, default='all',
                       choices=['all', 'even', 'odd'],
                       help='Which runs to extract from (default: all)')
    parser.add_argument('--space', type=str, default='MNI152NLin2009cAsym',
                       help='Analysis space')
    parser.add_argument('--derivatives-dir', type=str,
                       default='/work/upschrimpf1/mehrer/datasets/Marvi_2025_efficient_fMRI_localizer/derivatives',
                       help='Derivatives directory')
    parser.add_argument('--output', type=str,
                       help='Output CSV file (default: auto-generate)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ROI Response Extraction")
    print("="*70)
    print(f"Subject: {args.subject}")
    print(f"ROI mask: {args.roi_mask}")
    print(f"Extract from: {args.extract_from} runs")
    if args.contrasts:
        print(f"Contrasts: {', '.join(args.contrasts)}")
    print("="*70)
    
    # Initialize extractor
    extractor = ROIResponseExtractor(
        derivatives_dir=args.derivatives_dir,
        subject_id=args.subject,
        space=args.space
    )
    
    # Extract responses
    responses_df = extractor.extract_roi_responses(
        roi_mask_path=Path(args.roi_mask),
        contrasts=args.contrasts,
        run_split=args.extract_from
    )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        roi_name = Path(args.roi_mask).stem
        output_dir = Path(args.derivatives_dir) / args.subject / 'roi_responses'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{args.subject}_{roi_name}_responses.csv'
    
    responses_df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print("Extraction Complete!")
    print("="*70)
    print(f"Responses: {len(responses_df)}")
    print(f"Contrasts: {len(responses_df['contrast'].unique())}")
    print(f"Runs: {len(responses_df['run'].unique())}")
    print(f"Output: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()


