#!/usr/bin/env python3
"""
Cross-Validation Framework for fROI Definition

This script implements the cross-validation methodology from Marvi et al. (2025):
1. Define fROIs on odd runs (001, 003, 005)
2. Extract responses from even runs (002, 004)
3. Define fROIs on even runs (002, 004)
4. Extract responses from odd runs (001, 003, 005)
5. Compute reliability metrics:
   - Spatial overlap (Dice coefficient)
   - Response correlation
   - Selectivity consistency

This provides independent validation that fROIs generalize across data splits.

Usage:
------
# Single ROI cross-validation
python 15_roi_cross_validation.py --subject sub-kaneff01 --parcel-name ffa --hemisphere lh

# Multiple ROIs
python 15_roi_cross_validation.py --subject sub-kaneff01 --parcel-category julian --all-parcels
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import nibabel as nib
import pandas as pd
import json
from scipy.stats import pearsonr

# Import our tools from package
from emfl.roi.definition import fROIDefiner
from emfl.roi.extraction import ROIResponseExtractor  
from emfl.config import PARCEL_CONTRAST_MAP


class CrossValidationAnalyzer:
    """
    Perform cross-validation analysis for fROI reliability.
    
    Parameters
    ----------
    parcels_dir : Path
        Directory containing anatomical parcels
    derivatives_dir : Path
        fMRIprep derivatives directory
    subject_id : str
        Subject identifier
    space : str
        Analysis space
    percentile : float
        Top N% of voxels to select for fROI
    """
    
    def __init__(self, parcels_dir: Path, derivatives_dir: Path,
                 subject_id: str, space: str = 'MNI152NLin2009cAsym',
                 percentile: float = 10.0):
        self.parcels_dir = Path(parcels_dir)
        self.derivatives_dir = Path(derivatives_dir)
        self.subject_id = subject_id
        self.space = space
        self.percentile = percentile
        
        # Initialize tools
        self.definer = fROIDefiner(parcels_dir, derivatives_dir, subject_id, space)
        self.extractor = ROIResponseExtractor(derivatives_dir, subject_id, space)
    
    def compute_dice_coefficient(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute Dice coefficient between two binary masks.
        
        Dice = 2 * |A ∩ B| / (|A| + |B|)
        
        Parameters
        ----------
        mask1, mask2 : np.ndarray
            Binary masks
            
        Returns
        -------
        float
            Dice coefficient (0-1)
        """
        mask1_bool = mask1 > 0
        mask2_bool = mask2 > 0
        
        intersection = np.sum(mask1_bool & mask2_bool)
        size1 = np.sum(mask1_bool)
        size2 = np.sum(mask2_bool)
        
        if size1 + size2 == 0:
            return 0.0
        
        dice = 2.0 * intersection / (size1 + size2)
        return float(dice)
    
    def load_existing_froi(self, parcel_category: str, parcel_name: str,
                          hemisphere: str = None, run_split: str = 'even') -> tuple:
        """
        Load existing fROI mask from Task 8.4.
        
        Returns
        -------
        tuple
            (mask_path, n_voxels)
        """
        # Build path to existing mask from Task 8.4
        hemi_str = f"_{hemisphere}" if hemisphere else ""
        mask_dir = self.derivatives_dir / self.subject_id / 'frois' / f"{parcel_category}_{parcel_name}"
        mask_file = mask_dir / f"{self.subject_id}_{parcel_name}{hemi_str}_space-{self.space}_split-{run_split}_froi.nii.gz"
        
        if not mask_file.exists():
            raise FileNotFoundError(f"fROI mask not found: {mask_file}")
        
        # Load mask and count voxels
        mask_img = nib.load(mask_file)
        n_voxels = int(np.sum(mask_img.get_fdata() > 0))
        
        return str(mask_file), n_voxels
    
    def cross_validate_roi(self, parcel_category: str, parcel_name: str,
                          hemisphere: str = None) -> Dict:
        """
        Perform cross-validation for a single ROI.
        
        Loads pre-defined fROI masks from Task 8.4 (both even and odd splits),
        extracts responses from complementary runs, and computes reliability metrics.
        
        Parameters
        ----------
        parcel_category : str
            Category of parcel (e.g., 'julian', 'language')
        parcel_name : str
            Name of the parcel (e.g., 'ffa', 'ppa')
        hemisphere : str, optional
            Hemisphere ('lh' or 'rh')
            
        Returns
        -------
        Dict
            Cross-validation results with metrics
        """
        roi_label = f"{hemisphere}_{parcel_name}" if hemisphere else parcel_name
        
        print("\n" + "="*70)
        print(f"Cross-Validation: {roi_label}")
        print("="*70)
        
        results = {
            'subject': self.subject_id,
            'parcel_category': parcel_category,
            'parcel_name': parcel_name,
            'hemisphere': hemisphere,
            'roi_label': roi_label,
            'percentile': self.percentile,
            'space': self.space
        }
        
        # STEP 1: Load ODD-split fROI mask (from Task 8.4)
        print("\n[1/4] Loading ODD-split fROI mask (from Task 8.4)...")
        try:
            froi_odd_path, froi_odd_voxels = self.load_existing_froi(
                parcel_category=parcel_category,
                parcel_name=parcel_name,
                hemisphere=hemisphere,
                run_split='odd'
            )
            results['froi_odd_voxels'] = froi_odd_voxels
            results['froi_odd_path'] = froi_odd_path
            print(f"  + Loaded odd-split fROI: {froi_odd_voxels} voxels")
            print(f"    {Path(froi_odd_path).name}")
        except Exception as e:
            print(f"  x Error loading odd-split fROI: {e}")
            results['error'] = f"Odd-split fROI not found: {e}"
            return results
        
        # STEP 2: Extract responses from EVEN runs using odd-defined fROI
        print("\n[2/4] Extracting from EVEN runs (002, 004) using odd-split fROI...")
        try:
            responses_even = self.extractor.extract_roi_responses(
                roi_mask_path=froi_odd_path,
                run_split='even'
            )
            results['responses_even_from_odd'] = len(responses_even)
            print(f"  + Extracted {len(responses_even)} responses from even runs")
        except Exception as e:
            print(f"  x Error extracting even-run responses: {e}")
            results['error'] = f"Even-run extraction failed: {e}"
            return results
        
        # STEP 3: Load EVEN-split fROI mask (from Task 8.4)
        print("\n[3/4] Loading EVEN-split fROI mask (from Task 8.4)...")
        try:
            froi_even_path, froi_even_voxels = self.load_existing_froi(
                parcel_category=parcel_category,
                parcel_name=parcel_name,
                hemisphere=hemisphere,
                run_split='even'
            )
            results['froi_even_voxels'] = froi_even_voxels
            results['froi_even_path'] = str(froi_even_path)
            print(f"  + Loaded even-split fROI: {froi_even_voxels} voxels")
            print(f"    {Path(froi_even_path).name}")
        except Exception as e:
            print(f"  x Error loading even-split fROI: {e}")
            results['error'] = f"Even-split fROI not found: {e}"
            return results
        
        # STEP 4: Extract responses from ODD runs using even-defined fROI
        print("\n[4/4] Extracting from ODD runs (001, 003, 005) using even-split fROI...")
        try:
            responses_odd = self.extractor.extract_roi_responses(
                roi_mask_path=froi_even_path,
                run_split='odd'
            )
            results['responses_odd_from_even'] = len(responses_odd)
            print(f"  + Extracted {len(responses_odd)} responses from odd runs")
        except Exception as e:
            print(f"  x Error extracting odd-run responses: {e}")
            results['error'] = f"Odd-run extraction failed: {e}"
            return results
        
        # STEP 5: Compute reliability metrics
        print("\n[5/5] Computing reliability metrics...")
        
        # Dice coefficient (spatial overlap)
        froi_odd_img = nib.load(froi_odd_path)
        froi_even_img = nib.load(froi_even_path)
        froi_odd_data = froi_odd_img.get_fdata()
        froi_even_data = froi_even_img.get_fdata()
        dice = self.compute_dice_coefficient(froi_odd_data, froi_even_data)
        results['dice_coefficient'] = float(dice)
        print(f"  Dice coefficient: {dice:.3f}")
        
        # Response correlation (for preferred contrast)
        contrast_name = PARCEL_CONTRAST_MAP.get(parcel_name)
        if contrast_name:
            # Get preferred contrast responses from both splits
            even_pref = responses_even[responses_even['contrast'] == contrast_name]['response'].values
            odd_pref = responses_odd[responses_odd['contrast'] == contrast_name]['response'].values
            
            if len(even_pref) > 0 and len(odd_pref) > 0:
                # Align by run order (even has 2 runs, odd has 3 runs)
                # We'll use mean for simplicity
                even_mean = np.mean(even_pref)
                odd_mean = np.mean(odd_pref)
                
                results['even_mean_response'] = float(even_mean)
                results['odd_mean_response'] = float(odd_mean)
                results['mean_difference'] = float(abs(even_mean - odd_mean))
                
                print(f"  Preferred contrast ({contrast_name}):")
                print(f"    Even runs: {even_mean:.3f}")
                print(f"    Odd runs: {odd_mean:.3f}")
                print(f"    Difference: {abs(even_mean - odd_mean):.3f}")
        
        # Voxel-wise spatial pattern correlation (as in paper's Supp. Fig. S8)
        # Correlate z-score patterns across voxels within anatomical parcel
        print("\n[5/5] Computing voxel-wise spatial pattern correlation...")
        try:
            spatial_corr, spatial_pval = self.compute_spatial_pattern_correlation(
                parcel_category=parcel_category,
                parcel_name=parcel_name,
                hemisphere=hemisphere,
                contrast_name=contrast_name
            )
            results['spatial_correlation'] = float(spatial_corr)
            results['spatial_pval'] = float(spatial_pval)
            print(f"  Spatial pattern correlation: r={spatial_corr:.3f}, p={spatial_pval:.4f}")
        except Exception as e:
            print(f"  x Could not compute spatial correlation: {e}")
            results['spatial_correlation'] = None
            results['spatial_pval'] = None
        
        # Save response data
        output_dir = self.derivatives_dir / self.subject_id / 'roi_cross_validation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        responses_even_path = output_dir / f'{self.subject_id}_{roi_label}_even_from_odd.csv'
        responses_odd_path = output_dir / f'{self.subject_id}_{roi_label}_odd_from_even.csv'
        
        responses_even.to_csv(responses_even_path, index=False)
        responses_odd.to_csv(responses_odd_path, index=False)
        
        results['responses_even_path'] = str(responses_even_path)
        results['responses_odd_path'] = str(responses_odd_path)
        
        print("\n" + "="*70)
        print("Cross-Validation Complete!")
        print("="*70)
        print(f"Spatial overlap (Dice): {dice:.3f}")
        if 'spatial_correlation' in results and results['spatial_correlation'] is not None:
            print(f"Spatial pattern correlation: {results['spatial_correlation']:.3f}")
        print("="*70)
        
        return results
    
    def compute_spatial_pattern_correlation(self, parcel_category: str, parcel_name: str,
                                           hemisphere: str, contrast_name: str) -> tuple:
        """
        Compute voxel-wise spatial pattern correlation (Marvi et al. 2025, Supp. Fig. S8).
        
        This correlates the z-score pattern across ALL voxels in the anatomical parcel
        between even and odd run splits for the defining contrast.
        
        Parameters
        ----------
        parcel_category : str
            Parcel category
        parcel_name : str
            Parcel name
        hemisphere : str
            Hemisphere ('lh' or 'rh')
        contrast_name : str
            Defining contrast (e.g., 'faces_vs_objects')
            
        Returns
        -------
        tuple
            (correlation, p-value)
        """
        from emfl.roi.definition import fROIDefiner
        
        # Load anatomical parcel (not just the fROI mask)
        definer = fROIDefiner(
            parcels_dir=self.parcels_dir,
            derivatives_dir=self.derivatives_dir,
            subject_id=self.subject_id,
            space=self.space
        )
        
        parcel_img = definer.load_parcel(parcel_category, parcel_name, hemisphere)
        parcel_data = parcel_img.get_fdata()
        parcel_mask = parcel_data > 0
        
        # Determine modality
        visual_contrasts = ['faces_vs_objects', 'scenes_vs_objects', 'bodies_vs_objects',
                          'words_vs_objects', 'objects_vs_words']
        modality = 'visual' if contrast_name in visual_contrasts else 'auditory'
        
        # Load z-maps for defining contrast from even and odd splits
        glm_dir = self.derivatives_dir / self.subject_id / 'first_level_glm'
        
        # Even-split z-maps (averaged across even runs)
        even_split_dir = glm_dir / f'effloc_{modality}_split-even'
        even_zmaps = []
        for run in ['002', '004']:
            if 'MNI' in self.space:
                zmap_file = even_split_dir / f'run-{run}' / f'{self.subject_id}_task-effloc_run-{run}_{modality}_{contrast_name}_space-{self.space}_res-2_zmap.nii.gz'
            else:
                zmap_file = even_split_dir / f'run-{run}' / f'{self.subject_id}_task-effloc_run-{run}_{modality}_{contrast_name}_space-{self.space}_zmap.nii.gz'
            
            if zmap_file.exists():
                even_zmaps.append(nib.load(zmap_file).get_fdata())
        
        # Odd-split z-maps (averaged across odd runs)
        odd_split_dir = glm_dir / f'effloc_{modality}_split-odd'
        odd_zmaps = []
        for run in ['001', '003', '005']:
            if 'MNI' in self.space:
                zmap_file = odd_split_dir / f'run-{run}' / f'{self.subject_id}_task-effloc_run-{run}_{modality}_{contrast_name}_space-{self.space}_res-2_zmap.nii.gz'
            else:
                zmap_file = odd_split_dir / f'run-{run}' / f'{self.subject_id}_task-effloc_run-{run}_{modality}_{contrast_name}_space-{self.space}_zmap.nii.gz'
            
            if zmap_file.exists():
                odd_zmaps.append(nib.load(zmap_file).get_fdata())
        
        if len(even_zmaps) == 0 or len(odd_zmaps) == 0:
            raise ValueError(f"Could not find z-maps for {contrast_name}")
        
        # Average z-maps within each split
        even_zmap_avg = np.mean(even_zmaps, axis=0)
        odd_zmap_avg = np.mean(odd_zmaps, axis=0)
        
        # Resample parcel to functional space if needed
        if parcel_mask.shape != even_zmap_avg.shape:
            from nilearn.image import resample_to_img
            parcel_resampled = resample_to_img(
                parcel_img,
                nib.Nifti1Image(even_zmap_avg, nib.load(
                    even_split_dir / f'run-002' / f'{self.subject_id}_task-effloc_run-002_{modality}_{contrast_name}_space-{self.space}_res-2_zmap.nii.gz'
                    if 'MNI' in self.space else
                    even_split_dir / f'run-002' / f'{self.subject_id}_task-effloc_run-002_{modality}_{contrast_name}_space-{self.space}_zmap.nii.gz'
                ).affine),
                interpolation='nearest'
            )
            parcel_mask = parcel_resampled.get_fdata() > 0
        
        # Extract z-scores for all voxels within parcel
        even_zscores_in_parcel = even_zmap_avg[parcel_mask]
        odd_zscores_in_parcel = odd_zmap_avg[parcel_mask]
        
        # Correlate z-score patterns
        corr, pval = pearsonr(even_zscores_in_parcel, odd_zscores_in_parcel)
        
        print(f"    Parcel voxels: {np.sum(parcel_mask)}")
        print(f"    Even z-scores: mean={np.mean(even_zscores_in_parcel):.3f}, std={np.std(even_zscores_in_parcel):.3f}")
        print(f"    Odd z-scores: mean={np.mean(odd_zscores_in_parcel):.3f}, std={np.std(odd_zscores_in_parcel):.3f}")
        
        return corr, pval


def main():
    parser = argparse.ArgumentParser(
        description='Cross-validation framework for fROI reliability',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--subject', type=str, required=True,
                       help='Subject ID')
    parser.add_argument('--parcel-category', type=str, default='julian',
                       choices=['julian', 'language', 'tom', 'md', 'speech', 'vwfa'],
                       help='Parcel category')
    parser.add_argument('--parcel-name', type=str,
                       help='Specific parcel name (e.g., ffa, ppa)')
    parser.add_argument('--hemisphere', type=str, choices=['lh', 'rh'],
                       help='Hemisphere (if applicable)')
    parser.add_argument('--all-parcels', action='store_true',
                       help='Process all parcels in the category')
    parser.add_argument('--percentile', type=float, default=10.0,
                       help='Top N%% of voxels to select (default: 10)')
    parser.add_argument('--space', type=str, default='MNI152NLin2009cAsym',
                       help='Analysis space')
    parser.add_argument('--parcels-dir', type=str,
                       default='src/aux/emfl_analysis-main/PARCELS',
                       help='Directory containing parcels')
    parser.add_argument('--derivatives-dir', type=str,
                       default='/work/upschrimpf1/mehrer/datasets/Marvi_2025_efficient_fMRI_localizer/derivatives',
                       help='Derivatives directory')
    parser.add_argument('--output', type=str,
                       help='Output summary CSV file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ROI Cross-Validation Analysis")
    print("="*70)
    print(f"Subject: {args.subject}")
    print(f"Parcel category: {args.parcel_category}")
    if args.all_parcels:
        print("Mode: All parcels in category")
    else:
        print(f"Parcel: {args.parcel_name}")
        if args.hemisphere:
            print(f"Hemisphere: {args.hemisphere}")
    print(f"Percentile: top {args.percentile}%")
    print(f"Space: {args.space}")
    print("="*70)
    
    # Initialize analyzer
    analyzer = CrossValidationAnalyzer(
        parcels_dir=args.parcels_dir,
        derivatives_dir=args.derivatives_dir,
        subject_id=args.subject,
        space=args.space,
        percentile=args.percentile
    )
    
    # Determine which parcels to process
    if args.all_parcels:
        # Get all parcels from category
        # For now, just do the main visual ones from julian
        if args.parcel_category == 'julian':
            parcels = [
                ('ffa', 'lh'), ('ffa', 'rh'),
                ('ppa', 'lh'), ('ppa', 'rh'),
                ('eba', 'lh'), ('eba', 'rh'),
                ('loc', 'lh'), ('loc', 'rh'),
                ('vwfa', 'lh')
            ]
        else:
            print(f"Error: --all-parcels not yet implemented for {args.parcel_category}")
            sys.exit(1)
    else:
        if not args.parcel_name:
            print("Error: Must specify --parcel-name or use --all-parcels")
            sys.exit(1)
        parcels = [(args.parcel_name, args.hemisphere)]
    
    # Process each parcel
    all_results = []
    for parcel_name, hemisphere in parcels:
        try:
            result = analyzer.cross_validate_roi(
                parcel_category=args.parcel_category,
                parcel_name=parcel_name,
                hemisphere=hemisphere
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nx Error processing {hemisphere}_{parcel_name}: {e}")
            all_results.append({
                'subject': args.subject,
                'parcel_name': parcel_name,
                'hemisphere': hemisphere,
                'error': str(e)
            })
    
    # Save summary
    results_df = pd.DataFrame(all_results)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(args.derivatives_dir) / args.subject / 'roi_cross_validation'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{args.subject}_cross_validation_summary.csv'
    
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(f"ROIs processed: {len(all_results)}")
    print(f"Successful: {len([r for r in all_results if 'error' not in r])}")
    if 'dice_coefficient' in results_df.columns:
        print(f"\nMean Dice coefficient: {results_df['dice_coefficient'].mean():.3f}")
        print(f"Range: {results_df['dice_coefficient'].min():.3f} - {results_df['dice_coefficient'].max():.3f}")
    print(f"\nSummary saved to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()


