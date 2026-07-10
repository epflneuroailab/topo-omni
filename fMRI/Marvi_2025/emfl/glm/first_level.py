#!/usr/bin/env python3
"""
First-Level GLM Analysis for Marvi et al. 2025 EMFL Data

This script performs first-level (single-subject, single-run) GLM analysis
on preprocessed fMRI data to identify functional ROIs using the EMFL localizer.

The Efficient Multi-Functional Localizer (EMFL) presents both visual and auditory
stimuli simultaneously, allowing identification of multiple functional regions
in a single scanning session.

Key Contrasts:
--------------
Visual (from efflocVisualConditions):
    - Faces > Objects (FFA - Fusiform Face Area)
    - Scenes > Objects (PPA - Parahippocampal Place Area)
    - Bodies > Objects (EBA - Extrastriate Body Area)
    - Words > Objects (VWFA - Visual Word Form Area)
    - Objects > Words (LOC - Lateral Occipital Complex)

Auditory (from efflocAuditoryConditions):
    - False belief > False photo (ToM - Theory of Mind)
    - (False belief + False photo) > Nonwords (Language/Social - also known as English > Nonwords)
    - Nonwords > Quilted speech (Speech processing)
    - Math > (False belief + False photo) (Math/reasoning)
    - Math > Nonwords (Math vs language)
    - Math > Quilted (Math vs low-level speech)
    - English > Nonwords (Language comprehension - same as ToM > Nonwords)

References:
-----------
Marvi et al. (2025). An efficient multi-functional localizer task for
functional MRI research.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm import OLSModel
from nilearn import masking, image
from nilearn.plotting import plot_design_matrix, plot_stat_map, plot_glass_brain

# Import our event file parser from package
from emfl.io.events import (
    get_effloc_events,
    get_standard_task_events,
    create_design_matrix_inputs,
)


class EFMLOCFirstLevelGLM:
    """
    First-level GLM analysis for EMFL localizer data.
    
    Supports both volumetric and surface-based analysis in multiple spaces.
    
    Parameters
    ----------
    derivatives_dir : Path or str
        Path to fMRIprep derivatives directory
    subject_id : str
        Subject identifier (e.g., 'sub-kaneff01')
    space : str
        Output space for analysis:
        - 'MNI152NLin2009cAsym': Standard MNI volumetric space
        - 'T1w': Native subject volumetric space
        - 'fsnative': FreeSurfer native surface space
        - 'fsaverage5': FreeSurfer template surface space (fsaverage5)
    smoothing_fwhm : float
        Smoothing kernel FWHM in mm (for volumetric) or vertices (for surface).
        Default: 3.0 mm (matching original Marvi et al. 2025 analysis)
    high_pass : float
        High-pass filter cutoff in Hz
    run_split : str
        Type of run split for cross-validation:
        - 'all': Use all runs (default)
        - 'even': Use only even-numbered runs (002, 004, etc.)
        - 'odd': Use only odd-numbered runs (001, 003, 005, etc.)
    
    Notes
    -----
    GLM parameters are set to match the original Marvi et al. (2025) FS-FAST analysis:
    - 3mm FWHM smoothing (original used 3mm)
    - Linear drift removal (polynomial order 1)
    - 6 motion confounds (translation + rotation, no derivatives or global signal)
    - SPM canonical HRF
    - AR(1) noise model
    
    Run splitting enables cross-validation:
    - Define fROIs on odd runs, validate on even runs
    - Define fROIs on even runs, validate on odd runs
    - Average results for robust estimates
    """
    
    def __init__(self, derivatives_dir: str, subject_id: str,
                 space: str = 'T1w',  # Native anatomical space (Marvi et al. 2025)
                 smoothing_fwhm: float = 3.0,
                 high_pass: float = 0.01,
                 run_split: str = 'all',
                 orig_data_dir: str = None):
        # RELEASE PORT NOTE (docs/DESIGN.md §7 / README §6b): the events TSVs live in the
        # RAW BIDS tree, not the derivatives. The dev engine hard-derived that tree by
        # `str(derivatives_dir).replace('derivatives', 'orig_data')` — a brittle sibling-
        # path hack. Here `orig_data_dir` makes the events/raw root explicit (the release
        # `--raw-root`). When None we fall back to the dev replace-hack so existing dev
        # layouts still resolve unchanged (backward compatible).

        self.derivatives_dir = Path(derivatives_dir)
        self.subject_id = subject_id
        self.space = space
        self.smoothing_fwhm = smoothing_fwhm
        self.high_pass = high_pass
        self.run_split = run_split
        
        # Validate run_split
        if run_split not in ['all', 'even', 'odd']:
            raise ValueError(f"run_split must be 'all', 'even', or 'odd', got: {run_split}")
        
        # Determine if this is surface or volumetric space
        self.is_surface = 'fs' in space.lower()
        
        # Validate space
        valid_spaces = ['MNI152NLin2009cAsym', 'T1w', 'fsnative', 'fsaverage5', 'fsaverage6']
        if space not in valid_spaces:
            raise ValueError(f"Space must be one of {valid_spaces}, got: {space}")
        
        self.subject_dir = self.derivatives_dir / subject_id
        self.func_dir = self.subject_dir / 'func'
        
        # Get original (raw BIDS) data directory for events. Explicit root when given;
        # else the dev sibling-path fallback (see RELEASE PORT NOTE above).
        if orig_data_dir is not None:
            self.orig_data_dir = Path(orig_data_dir)
        else:
            self.orig_data_dir = Path(str(derivatives_dir).replace('derivatives', 'orig_data'))
        self.orig_subject_dir = self.orig_data_dir / subject_id
        
        if not self.func_dir.exists():
            raise FileNotFoundError(f"Functional directory not found: {self.func_dir}")
        if not self.orig_subject_dir.exists():
            raise FileNotFoundError(f"Original subject directory not found: {self.orig_subject_dir}")
    
    def get_preprocessed_bold_path(self, task: str, run: str, hemi: str = None):
        """
        Get path to preprocessed BOLD file.
        
        Parameters
        ----------
        task : str
            Task name (e.g., 'effloc')
        run : str
            Run number (e.g., '001')
        hemi : str, optional
            Hemisphere for surface data ('L' or 'R'). Required for surface spaces.
            
        Returns
        -------
        Path or tuple of Paths
            For volumetric: single Path
            For surface: dict with 'L' and 'R' paths if hemi is None, else single Path
        """
        if self.is_surface:
            # Surface data (GIFTI files, separate L/R hemispheres)
            if hemi is not None:
                # Return specific hemisphere
                pattern = f"{self.subject_id}_task-{task}_run-{run}_hemi-{hemi}_space-{self.space}_bold.func.gii"
                bold_path = self.func_dir / pattern
                if not bold_path.exists():
                    raise FileNotFoundError(f"Preprocessed BOLD not found: {bold_path}")
                return bold_path
            else:
                # Return both hemispheres
                paths = {}
                for h in ['L', 'R']:
                    pattern = f"{self.subject_id}_task-{task}_run-{run}_hemi-{h}_space-{self.space}_bold.func.gii"
                    bold_path = self.func_dir / pattern
                    if not bold_path.exists():
                        raise FileNotFoundError(f"Preprocessed BOLD not found: {bold_path}")
                    paths[h] = bold_path
                return paths
        else:
            # Volumetric data (NIfTI files)
            if self.space == 'MNI152NLin2009cAsym':
                pattern = f"{self.subject_id}_task-{task}_run-{run}_space-{self.space}_res-2_desc-preproc_bold.nii.gz"
            else:
                pattern = f"{self.subject_id}_task-{task}_run-{run}_space-{self.space}_desc-preproc_bold.nii.gz"
            
            bold_path = self.func_dir / pattern
            if not bold_path.exists():
                raise FileNotFoundError(f"Preprocessed BOLD not found: {bold_path}")
            return bold_path
    
    def get_confounds_path(self, task: str, run: str) -> Path:
        """Get path to confounds file."""
        confounds_path = self.func_dir / f"{self.subject_id}_task-{task}_run-{run}_desc-confounds_timeseries.tsv"
        if not confounds_path.exists():
            raise FileNotFoundError(f"Confounds file not found: {confounds_path}")
        return confounds_path
    
    def load_confounds(self, task: str, run: str, 
                      confounds_to_use: List[str] = None) -> pd.DataFrame:
        """
        Load and select confounds for nuisance regression.
        
        Parameters
        ----------
        task : str
            Task name
        run : str
            Run number
        confounds_to_use : List[str], optional
            List of confound column names to use. If None, uses default set
            matching original Marvi et al. (2025) analysis (6 motion parameters only).
            
        Returns
        -------
        pd.DataFrame
            Selected confounds
            
        Notes
        -----
        Default confounds match the original FS-FAST analysis: 6 motion parameters
        (3 translations + 3 rotations) without derivatives or global signal.
        """
        if confounds_to_use is None:
            # Default confounds: 6 motion parameters only (matching original analysis)
            confounds_to_use = [
                'trans_x', 'trans_y', 'trans_z',
                'rot_x', 'rot_y', 'rot_z'
            ]
        
        confounds_path = self.get_confounds_path(task, run)
        confounds_df = pd.read_csv(confounds_path, sep='\t')
        
        # Select only requested confounds that exist
        available_confounds = [c for c in confounds_to_use if c in confounds_df.columns]
        confounds = confounds_df[available_confounds]
        
        # Fill NaN values (first timepoint for derivatives)
        confounds = confounds.fillna(0)
        
        return confounds
    
    def run_effloc_glm(self, run: str, modality: str = 'visual', 
                      save_outputs: bool = True):
        """
        Run first-level GLM for EMFL localizer.
        
        Handles both volumetric and surface data automatically based on the space.
        
        Parameters
        ----------
        run : str
            Run number (e.g., '001')
        modality : str
            'visual' or 'auditory'
        save_outputs : bool
            Whether to save contrast maps
            
        Returns
        -------
        For volumetric: (FirstLevelModel, Dict of contrast maps)
        For surface: Dict with 'L' and 'R' containing (FirstLevelModel, Dict) for each hemisphere
        """
        print(f"\n{'='*70}")
        print(f"Running EMFL GLM: {self.subject_id} | Run {run} | {modality.capitalize()}")
        print(f"Space: {self.space} ({'surface' if self.is_surface else 'volumetric'})")
        print(f"{'='*70}\n")
        
        if self.is_surface:
            # Process each hemisphere separately for surface data
            return self._run_effloc_glm_surface(run, modality, save_outputs)
        else:
            # Process volumetric data
            return self._run_effloc_glm_volumetric(run, modality, save_outputs)
    
    def _run_effloc_glm_volumetric(self, run: str, modality: str, save_outputs: bool):
        """Run GLM for volumetric data."""
        # Load preprocessed BOLD
        bold_path = self.get_preprocessed_bold_path('effloc', run)
        print(f"Loading BOLD: {bold_path.name}")
        bold_img = nib.load(bold_path)
        
        # Load events
        print(f"Loading {modality} events...")
        events = get_effloc_events(self.orig_subject_dir, run, modality=modality)
        print(f"  Found {len(events)} events across {len(events['trial_type'].unique())} conditions")
        
        # Load confounds
        print("Loading confounds...")
        confounds = self.load_confounds('effloc', run)
        print(f"  Using {len(confounds.columns)} confound regressors")
        
        # Get TR
        tr = float(bold_img.header.get_zooms()[3])
        print(f"  TR = {tr:.3f}s")
        
        # Create and fit GLM model
        print("\nFitting GLM model...")
        # Parameters match original Marvi et al. (2025) FS-FAST analysis:
        # - drift_model='polynomial' for linear drift (order 1, matching FS-FAST polyfit 1)
        # - smoothing_fwhm=3.0mm, hrf_model='spm', noise_model='ar1'
        glm = FirstLevelModel(
            t_r=tr,
            noise_model='ar1',
            standardize=False,
            hrf_model='spm',
            drift_model='polynomial',
            drift_order=1,
            high_pass=self.high_pass,
            smoothing_fwhm=self.smoothing_fwhm,
            minimize_memory=False
        )
        
        glm.fit(bold_img, events=events, confounds=confounds)
        print("  GLM fitting complete!")
        
        # Compute per-condition effect maps (for Figure 4-style plots)
        print("\nComputing per-condition effect maps...")
        condition_maps = self._compute_condition_effects(glm, events, modality)
        
        # Compute contrasts
        print("\nComputing contrasts...")
        beta_maps, t_maps, p_value_maps = self._compute_contrasts_volumetric(glm, modality)
        # RELEASE PORT NOTE (docs/DESIGN.md §7 / README §6b): the dev volumetric path saved
        # only beta/tmap/pval + per-condition effect maps and *dropped the zmap* — but the
        # entire Branch-A downstream (fROI definition, cross-validation, response
        # extraction) reads `..._res-2_zmap.nii.gz`, and that is what the published cut's
        # split dirs actually contain. So the `--input-source raw` path was structurally
        # broken end-to-end. We restore the z-score contrast + zmap save here (mirroring
        # the surface path, which already saves zmap) so the regenerated cut feeds Branch A.
        z_maps = self._compute_zscore_contrasts(glm, modality)

        # Save outputs if requested
        if save_outputs:
            self._save_condition_maps(condition_maps, run, modality)
            self._save_contrast_maps(z_maps, run, modality, map_type='zmap')
            self._save_contrast_maps(beta_maps, run, modality, map_type='beta')
            self._save_contrast_maps(t_maps, run, modality, map_type='tmap')
            self._save_contrast_maps(p_value_maps, run, modality, map_type='pval')

        return glm, beta_maps
    
    def _run_effloc_glm_surface(self, run: str, modality: str, save_outputs: bool):
        """
        Run GLM for surface data (processes each hemisphere separately).
        
        Uses manual vertex-wise GLM fitting with OLSModel (following Hauptman 2024 approach)
        instead of FirstLevelModel which has limited GIFTI support.
        """
        results = {}
        
        # Load events and confounds (same for both hemispheres)
        print(f"Loading {modality} events...")
        events = get_effloc_events(self.orig_subject_dir, run, modality=modality)
        print(f"  Found {len(events)} events across {len(events['trial_type'].unique())} conditions")
        
        print("Loading confounds...")
        confounds = self.load_confounds('effloc', run)
        print(f"  Using {len(confounds.columns)} confound regressors")
        
        # Process each hemisphere
        for hemi in ['L', 'R']:
            print(f"\n--- Processing Hemisphere {hemi} ---")
            
            # Load GIFTI file
            bold_path = self.get_preprocessed_bold_path('effloc', run, hemi=hemi)
            print(f"Loading BOLD: {bold_path.name}")
            func_data_img = nib.load(bold_path)
            
            # Extract time series (vertices x time)
            # GIFTI darrays are in time x vertices format, we need vertices x time
            n_scans = len(func_data_img.darrays)
            time_series = np.array([darray.data for darray in func_data_img.darrays]).T
            n_vertices = time_series.shape[0]
            
            print(f"  Loaded {n_vertices} vertices × {n_scans} timepoints")
            
            # Get TR
            tr = 2.0  # Standard TR for this dataset
            print(f"  TR = {tr:.3f}s")
            
            # Create design matrix
            print(f"Creating design matrix for hemisphere {hemi}...")
            frame_times = np.arange(n_scans) * tr
            # Parameters match original Marvi et al. (2025) analysis
            design_matrix = make_first_level_design_matrix(
                frame_times,
                events,
                add_regs=confounds.values,
                add_reg_names=confounds.columns.tolist(),
                hrf_model='spm',
                drift_model='polynomial',
                drift_order=1,
                high_pass=self.high_pass
            )
            print(f"  Design matrix: {design_matrix.shape}")
            
            # Fit GLM vertex-by-vertex
            print(f"Fitting GLM for {n_vertices} vertices (hemisphere {hemi})...")
            betas = []
            residuals = []
            
            for vertex_idx in range(n_vertices):
                vertex_data = time_series[vertex_idx, :]
                glm = OLSModel(design_matrix.values)
                glm_fit = glm.fit(vertex_data)
                betas.append(glm_fit.theta)
                residuals.append(glm_fit.residuals)
                
                # Progress indicator
                if (vertex_idx + 1) % 5000 == 0:
                    print(f"    Processed {vertex_idx + 1}/{n_vertices} vertices...")
            
            betas = np.array(betas)  # Shape: (n_vertices, n_regressors)
            residuals = np.array(residuals)  # Shape: (n_vertices, n_scans)
            print(f"  GLM fitting complete for hemisphere {hemi}!")
            
            # Compute contrasts
            print(f"Computing contrasts for hemisphere {hemi}...")
            beta_maps, t_maps, p_value_maps = self._compute_contrasts_surface(
                betas, residuals, design_matrix, modality
            )
            
            results[hemi] = (design_matrix, betas, residuals, beta_maps, t_maps, p_value_maps)
        
        # Save outputs if requested
        if save_outputs:
            for hemi in ['L', 'R']:
                _, _, _, beta_maps, t_maps, p_value_maps = results[hemi]
                self._save_contrast_maps(beta_maps, run, modality, hemi=hemi, map_type='beta')
                self._save_contrast_maps(t_maps, run, modality, hemi=hemi, map_type='tmap')
                self._save_contrast_maps(p_value_maps, run, modality, hemi=hemi, map_type='pval')
        
        return results
    
    def _compute_contrasts(self, glm: FirstLevelModel, modality: str) -> Dict:
        """
        Compute contrasts for EMFL localizer (legacy method for old code compatibility).
        
        Parameters
        ----------
        glm : FirstLevelModel
            Fitted GLM model
        modality : str
            'visual' or 'auditory'
            
        Returns
        -------
        Dict
            Dictionary of contrast names and maps
        """
        contrast_maps = {}
        
        if modality == 'visual':
            # Visual contrasts (as specified in Marvi et al. 2025)
            contrasts = {
                'faces_vs_objects': 'faces - objects',
                'scenes_vs_objects': 'scenes - objects',
                'bodies_vs_objects': 'bodies - objects',
                'words_vs_objects': 'words_scr_objects - objects',
                'objects_vs_words': 'objects - words_scr_objects'
            }
        else:  # auditory
            # Auditory contrasts (matching original Marvi et al. 2025 MATLAB analysis)
            contrasts = {
                'false_belief_vs_false_photo': 'false_belief - false_photo',
                'nonwords_vs_quilted': 'nonwords - quilted_speech',
                'math_vs_theory_of_mind': 'math - 0.5*false_belief - 0.5*false_photo',
                'english_vs_nonwords': '0.5*false_belief + 0.5*false_photo - nonwords'
            }
        
        for contrast_name, contrast_formula in contrasts.items():
            try:
                print(f"  Computing: {contrast_name}")
                contrast_map = glm.compute_contrast(contrast_formula, output_type='z_score')
                contrast_maps[contrast_name] = contrast_map
            except Exception as e:
                print(f"  Warning: Could not compute {contrast_name}: {e}")
        
        return contrast_maps
    
    @staticmethod
    def _volumetric_contrast_formulas(modality: str) -> Dict:
        """The 5 visual / 4 auditory EMFL contrast formulas (Marvi et al. 2025 Table 3).

        Single source of truth for the volumetric contrast names + formulas, shared by
        `_compute_contrasts_volumetric` (beta/t/p) and `_compute_zscore_contrasts` (zmap)
        so every saved map type uses identical contrast names.
        """
        if modality == 'visual':
            return {
                'faces_vs_objects': 'faces - objects',
                'scenes_vs_objects': 'scenes - objects',
                'bodies_vs_objects': 'bodies - objects',
                'words_vs_objects': 'words_scr_objects - objects',
                'objects_vs_words': 'objects - words_scr_objects'
            }
        # auditory
        return {
            'false_belief_vs_false_photo': 'false_belief - false_photo',
            'nonwords_vs_quilted': 'nonwords - quilted_speech',
            'math_vs_theory_of_mind': 'math - 0.5*false_belief - 0.5*false_photo',
            'english_vs_nonwords': '0.5*false_belief + 0.5*false_photo - nonwords'
        }

    def _compute_zscore_contrasts(self, glm: FirstLevelModel, modality: str) -> Dict:
        """Compute z-score contrast maps (output_type='z_score') for the volumetric path.

        RELEASE PORT ADDITION (see the note in `_run_effloc_glm_volumetric`): the dev
        volumetric engine emitted only beta/tmap/pval, but Branch A reads `_res-2_zmap`.
        This mirrors the surface path's z-score contrasts so the raw-regenerated cut
        contains the zmaps that fROI definition / cross-validation / extraction require.
        """
        z_maps = {}
        for contrast_name, contrast_formula in self._volumetric_contrast_formulas(modality).items():
            try:
                z_maps[contrast_name] = glm.compute_contrast(contrast_formula, output_type='z_score')
            except Exception as e:  # noqa: BLE001 - match engine's per-contrast skip-on-error
                print(f"  Warning: Could not compute zmap {contrast_name}: {e}")
        return z_maps

    def _compute_contrasts_volumetric(self, glm: FirstLevelModel, modality: str):
        """
        Compute contrasts for volumetric data with beta, t-map, and p-value outputs.
        Similar to surface version, returns separate maps for each statistic type.
        
        Parameters
        ----------
        glm : FirstLevelModel
            Fitted GLM model
        modality : str
            'visual' or 'auditory'
            
        Returns
        -------
        Tuple[Dict, Dict, Dict]
            (beta_maps, t_maps, p_value_maps) - each a dict of contrast name -> map
        """
        beta_maps = {}
        t_maps = {}
        p_value_maps = {}

        contrasts = self._volumetric_contrast_formulas(modality)

        for contrast_name, contrast_formula in contrasts.items():
            try:
                print(f"  Computing: {contrast_name}")
                # Compute effect size (beta)
                beta_map = glm.compute_contrast(contrast_formula, output_type='effect_size')
                # Compute t-statistic
                t_map = glm.compute_contrast(contrast_formula, output_type='stat')
                # Compute p-value (one-tailed, positive direction)
                p_map = glm.compute_contrast(contrast_formula, output_type='p_value')
                
                beta_maps[contrast_name] = beta_map
                t_maps[contrast_name] = t_map
                p_value_maps[contrast_name] = p_map
                
            except Exception as e:
                print(f"  Warning: Could not compute {contrast_name}: {e}")
        
        return beta_maps, t_maps, p_value_maps
    
    def _compute_condition_effects(self, glm: FirstLevelModel, events: pd.DataFrame, modality: str) -> Dict:
        """
        Compute effect size maps for each individual condition.
        
        This replicates the paper's approach of extracting per-condition beta estimates
        (Figure 4, Figure 6). Effect size maps are roughly equivalent to percent signal change.
        
        Parameters
        ----------
        glm : FirstLevelModel
            Fitted GLM model
        events : pd.DataFrame
            Events DataFrame with trial_type column
        modality : str
            'visual' or 'auditory'
            
        Returns
        -------
        Dict
            Dictionary of condition name -> effect map
        """
        from emfl.io.events import get_unique_conditions
        
        condition_maps = {}
        
        # Get unique conditions from events
        conditions = get_unique_conditions(events)
        
        print(f"  Computing effect maps for {len(conditions)} conditions...")
        for condition in conditions:
            try:
                # Compute effect size map (equivalent to beta / % signal change)
                effect_map = glm.compute_contrast(condition, output_type='effect_size')
                condition_maps[condition] = effect_map
                print(f"    ✓ {condition}")
            except Exception as e:
                print(f"    ✗ {condition}: {e}")
        
        return condition_maps
    
    def _save_condition_maps(self, condition_maps: Dict, run: str, modality: str, hemi: str = None):
        """
        Save per-condition effect maps.
        
        Parameters
        ----------
        condition_maps : Dict
            Dictionary of condition name -> effect map
        run : str
            Run identifier
        modality : str
            'visual' or 'auditory'
        hemi : str, optional
            Hemisphere ('L' or 'R') for surface data
        """
        # Create output directory (same structure as contrast maps)
        if self.run_split == 'all':
            output_dir = self.subject_dir / 'first_level_glm' / f'effloc_{modality}' / f'run-{run}'
        else:
            output_dir = self.subject_dir / 'first_level_glm' / f'effloc_{modality}_split-{self.run_split}' / f'run-{run}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving {len(condition_maps)} condition effect maps...")
        
        for condition_name, effect_map in condition_maps.items():
            # Build filename
            if self.is_surface:
                # Surface: save as GIFTI
                filename = f'{self.subject_id}_task-effloc_run-{run}_{modality}_{condition_name}_hemi-{hemi}_space-{self.space}_effect.func.gii'
            else:
                # Volumetric: save as NIfTI
                if 'MNI' in self.space:
                    filename = f'{self.subject_id}_task-effloc_run-{run}_{modality}_{condition_name}_space-{self.space}_res-2_effect.nii.gz'
                else:
                    filename = f'{self.subject_id}_task-effloc_run-{run}_{modality}_{condition_name}_space-{self.space}_effect.nii.gz'
            
            output_path = output_dir / filename
            nib.save(effect_map, output_path)
        
        print(f"  Saved to: {output_dir}")
    
    def _compute_contrasts_surface(self, betas: np.ndarray, residuals: np.ndarray, 
                                   design_matrix: pd.DataFrame, modality: str) -> tuple:
        """
        Compute contrasts for surface data using manual t-map calculation.
        
        Follows the approach from Hauptman 2024 GLM analysis.
        
        Parameters
        ----------
        betas : np.ndarray
            Beta coefficients from GLM fit (n_vertices, n_regressors)
        residuals : np.ndarray
            Residuals from GLM fit (n_vertices, n_scans)
        design_matrix : pd.DataFrame
            Design matrix used for GLM
        modality : str
            'visual' or 'auditory'
            
        Returns
        -------
        tuple (Dict, Dict, Dict)
            (beta_maps, t_maps, p_value_maps) - dictionaries of contrast name -> map
        """
        from scipy import stats as scipy_stats
        
        beta_maps = {}  # Effect size maps
        t_maps = {}  # T-statistic maps
        p_value_maps = {}  # P-value maps
        
        # Calculate degrees of freedom for t-distribution
        n_scans = residuals.shape[1]
        n_regressors = len(design_matrix.columns)
        df = n_scans - n_regressors
        
        # Get regressor names from design matrix
        regressor_names = design_matrix.columns.tolist()
        
        # Define contrasts based on modality
        if modality == 'visual':
            contrasts_to_compute = {
                'faces_vs_objects': ('faces', 'objects'),
                'scenes_vs_objects': ('scenes', 'objects'),
                'bodies_vs_objects': ('bodies', 'objects'),
                'words_vs_objects': ('words_scr_objects', 'objects'),
                'objects_vs_words': ('objects', 'words_scr_objects')
            }
        elif modality == 'auditory':
            contrasts_to_compute = {
                'false_belief_vs_false_photo': ('false_belief', 'false_photo'),
                'nonwords_vs_quilted': ('nonwords', 'quilted_speech'),
                'math_vs_theory_of_mind': ('math', ('false_belief', 'false_photo')),
                'english_vs_nonwords': (('false_belief', 'false_photo'), 'nonwords')
            }
        
        # Compute each contrast
        for contrast_name, (pos_conditions, neg_conditions) in contrasts_to_compute.items():
            try:
                print(f"  Computing: {contrast_name}")
                
                # Build contrast vector
                contrast_vector = np.zeros(len(regressor_names))
                
                # Positive conditions
                if isinstance(pos_conditions, str):
                    pos_conditions = [pos_conditions]
                for cond in pos_conditions:
                    if cond in regressor_names:
                        idx = regressor_names.index(cond)
                        contrast_vector[idx] = 1.0 / len(pos_conditions)
                
                # Negative conditions
                if isinstance(neg_conditions, str):
                    neg_conditions = [neg_conditions]
                for cond in neg_conditions:
                    if cond in regressor_names:
                        idx = regressor_names.index(cond)
                        contrast_vector[idx] = -1.0 / len(neg_conditions)
                
                # Compute contrast map
                contrast_map = np.dot(betas, contrast_vector)  # (n_vertices,)
                
                # Compute standard error for t-statistic
                residual_variance = np.var(residuals, axis=1, ddof=len(regressor_names))
                design_matrix_projection = np.dot(
                    np.dot(contrast_vector, np.linalg.pinv(design_matrix.T @ design_matrix)),
                    contrast_vector.T
                )
                standard_error = np.sqrt(residual_variance * design_matrix_projection)
                
                # Avoid division by zero
                standard_error[standard_error == 0] = np.nan
                standard_error = np.nan_to_num(standard_error, nan=np.inf)
                
                # Compute t-map
                t_map = contrast_map / standard_error
                
                # Compute p-values from t-map (one-tailed, positive direction)
                # Using survival function (1 - CDF) for right tail test
                p_map = scipy_stats.t.sf(t_map, df=df)
                
                # Store all map types
                beta_maps[contrast_name] = contrast_map  # Effect size (beta coefficients)
                t_maps[contrast_name] = t_map  # T-statistics
                p_value_maps[contrast_name] = p_map  # P-values
                
            except Exception as e:
                print(f"  Warning: Could not compute {contrast_name}: {e}")
        
        return beta_maps, t_maps, p_value_maps
    
    def _save_contrast_maps(self, contrast_maps: Dict, run: str, modality: str, hemi: str = None, map_type: str = 'zmap'):
        """
        Save contrast maps to disk.
        
        Parameters
        ----------
        contrast_maps : Dict
            Dictionary of contrast name -> contrast map
        run : str
            Run number (e.g., '001')
        modality : str
            'visual' or 'auditory'
        hemi : str, optional
            Hemisphere ('L' or 'R') for surface data. None for volumetric.
        map_type : str, optional
            Type of map: 'beta', 'tmap', 'pval', or 'zmap' (default)
        """
        # Include run_split in output directory if not 'all'
        if self.run_split == 'all':
            output_dir = self.subject_dir / 'first_level_glm' / f'effloc_{modality}' / f'run-{run}'
        else:
            output_dir = self.subject_dir / 'first_level_glm' / f'effloc_{modality}_split-{self.run_split}' / f'run-{run}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for contrast_name, contrast_map in contrast_maps.items():
            if hemi is not None:
                # Surface data - contrast_map is a numpy array, need to create GIFTI
                output_path = output_dir / f"{self.subject_id}_task-effloc_run-{run}_hemi-{hemi}_{modality}_{contrast_name}_space-{self.space}_{map_type}.func.gii"
                
                # Create GIFTI image from numpy array
                # Set appropriate intent based on map type
                if map_type == 'pval':
                    intent = 'NIFTI_INTENT_PVAL'
                elif map_type == 'tmap':
                    intent = 'NIFTI_INTENT_TTEST'
                elif map_type == 'beta':
                    intent = 'NIFTI_INTENT_ESTIMATE'
                else:  # zmap
                    intent = 'NIFTI_INTENT_ZSCORE'
                
                darray = nib.gifti.GiftiDataArray(
                    data=contrast_map.astype(np.float32),
                    intent=intent,
                    datatype='NIFTI_TYPE_FLOAT32'
                )
                gii_img = nib.gifti.GiftiImage(darrays=[darray])
                nib.save(gii_img, output_path)
            else:
                # Volumetric data - contrast_map is already a nibabel image
                if self.space == 'MNI152NLin2009cAsym':
                    output_path = output_dir / f"{self.subject_id}_task-effloc_run-{run}_{modality}_{contrast_name}_space-{self.space}_res-2_{map_type}.nii.gz"
                else:
                    output_path = output_dir / f"{self.subject_id}_task-effloc_run-{run}_{modality}_{contrast_name}_space-{self.space}_{map_type}.nii.gz"
                
                nib.save(contrast_map, output_path)
            
            print(f"  Saved: {output_path.name}")
    
    def visualize_contrast(self, contrast_map, title: str, output_path: Optional[Path] = None):
        """
        Visualize a contrast map.
        
        Parameters
        ----------
        contrast_map : Nifti1Image
            Contrast map to visualize
        title : str
            Plot title
        output_path : Path, optional
            Path to save figure
        """
        from matplotlib import pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Glass brain plot
        plot_glass_brain(contrast_map, axes=axes[0], colorbar=True,
                        threshold=2.3, title=f"{title} (Glass Brain)",
                        plot_abs=False, display_mode='lyrz')
        
        # Statistical map
        plot_stat_map(contrast_map, axes=axes[1], colorbar=True,
                     threshold=2.3, title=f"{title} (Axial Slices)",
                     cut_coords=5)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """
    Example usage: Run GLM analysis for a single subject and run.
    """
    # Configuration
    derivatives_dir = "/work/upschrimpf1/mehrer/datasets/Marvi_2025_efficient_fMRI_localizer/derivatives"
    subject_id = "sub-kaneff01"
    run = "001"
    
    print("="*70)
    print("EMFL First-Level GLM Analysis")
    print("="*70)
    print(f"Subject: {subject_id}")
    print(f"Run: {run}")
    print(f"Derivatives: {derivatives_dir}")
    print("="*70)
    
    # Initialize GLM analyzer
    analyzer = EFMLOCFirstLevelGLM(
        derivatives_dir=derivatives_dir,
        subject_id=subject_id,
        space='MNI152NLin2009cAsym',
        smoothing_fwhm=3.0,  # Updated to match original analysis (was 5.0)
        high_pass=0.01
    )
    
    # Run GLM for visual conditions
    print("\n" + "="*70)
    print("VISUAL CONDITIONS")
    print("="*70)
    glm_visual, contrasts_visual = analyzer.run_effloc_glm(run, modality='visual', save_outputs=True)
    
    # Run GLM for auditory conditions
    print("\n" + "="*70)
    print("AUDITORY CONDITIONS")
    print("="*70)
    glm_auditory, contrasts_auditory = analyzer.run_effloc_glm(run, modality='auditory', save_outputs=True)
    
    print("\n" + "="*70)
    print("GLM ANALYSIS COMPLETE")
    print("="*70)
    print(f"Visual contrasts: {list(contrasts_visual.keys())}")
    print(f"Auditory contrasts: {list(contrasts_auditory.keys())}")
    print(f"\nOutputs saved to: {analyzer.subject_dir / 'first_level_glm'}")


if __name__ == "__main__":
    main()

