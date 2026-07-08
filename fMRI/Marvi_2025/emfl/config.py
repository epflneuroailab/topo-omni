"""
Configuration Module for EMFL Analysis
======================================

Contains all shared constants, default parameters, and configuration options
used across the EMFL analysis pipeline.

This module centralizes configuration to:
- Reduce code duplication
- Make it easier to modify parameters globally
- Provide clear documentation of all constants
"""

from pathlib import Path

# =============================================================================
# Subject and Run Configuration
# =============================================================================

# Subjects with complete EMFL (effloc) data
ALL_SUBJECTS = [
    'sub-kaneff01',
    'sub-kaneff06',
    'sub-kaneff07',
    'sub-kaneff08',
    'sub-kaneff09',
    'sub-kaneff21'
]

# All EMFL runs (5 runs per subject)
ALL_RUNS = ['001', '002', '003', '004', '005']

# Run splits for cross-validation
RUN_SPLITS = {
    'all': ['001', '002', '003', '004', '005'],
    'even': ['002', '004'],
    'odd': ['001', '003', '005']
}

# =============================================================================
# GLM Default Parameters (aligned with Marvi et al. 2025)
# =============================================================================

# Smoothing kernel FWHM in mm (matches original FS-FAST analysis)
DEFAULT_SMOOTHING = 3.0

# High-pass filter cutoff in Hz
DEFAULT_HIGH_PASS = 0.01

# TR (repetition time) in seconds
DEFAULT_TR = 2.0

# Drift model for GLM (polynomial order 1 = linear drift removal)
DEFAULT_DRIFT_MODEL = 'polynomial'
DEFAULT_DRIFT_ORDER = 1

# Noise model
DEFAULT_NOISE_MODEL = 'ar1'

# Motion confound parameter names (6 motion parameters only)
MOTION_CONFOUNDS = [
    'trans_x', 'trans_y', 'trans_z',  # Translation
    'rot_x', 'rot_y', 'rot_z'          # Rotation
]

# =============================================================================
# Contrast Definitions
# =============================================================================

# Visual contrasts (6 total)
VISUAL_CONTRASTS = [
    'faces_vs_objects',
    'scenes_vs_objects',
    'bodies_vs_objects',
    'words_vs_objects',
    'objects_vs_words'
]

# Auditory contrasts (5 total)
AUDITORY_CONTRASTS = [
    'false_belief_vs_false_photo',
    'nonwords_vs_quilted',
    'math_vs_theory_of_mind',
    'english_vs_nonwords'
]

# All contrasts (9 total: 5 visual + 4 auditory, matches Marvi et al. 2025 paper)
ALL_CONTRASTS = VISUAL_CONTRASTS + AUDITORY_CONTRASTS

# Detailed contrast information for visualization
CONTRAST_INFO = {
    # Visual contrasts
    'faces_vs_objects': {
        'label': 'Faces > Objects',
        'roi': 'FFA (Fusiform Face Area)',
        'color_positive': 'Faces',
        'color_negative': 'Objects',
        'modality': 'visual'
    },
    'scenes_vs_objects': {
        'label': 'Scenes > Objects',
        'roi': 'PPA (Parahippocampal Place Area)',
        'color_positive': 'Scenes',
        'color_negative': 'Objects',
        'modality': 'visual'
    },
    'bodies_vs_objects': {
        'label': 'Bodies > Objects',
        'roi': 'EBA (Extrastriate Body Area)',
        'color_positive': 'Bodies',
        'color_negative': 'Objects',
        'modality': 'visual'
    },
    'words_vs_objects': {
        'label': 'Words > Objects',
        'roi': 'VWFA (Visual Word Form Area)',
        'color_positive': 'Words',
        'color_negative': 'Objects',
        'modality': 'visual'
    },
    'objects_vs_words': {
        'label': 'Objects > Words',
        'roi': 'LOC (Lateral Occipital Complex)',
        'color_positive': 'Objects',
        'color_negative': 'Words',
        'modality': 'visual'
    },
    # Auditory contrasts
    'false_belief_vs_false_photo': {
        'label': 'False Belief > False Photo',
        'roi': 'ToM (Theory of Mind - rTPJ)',
        'color_positive': 'False Belief',
        'color_negative': 'False Photo',
        'modality': 'auditory'
    },
    'nonwords_vs_quilted': {
        'label': 'Nonwords > Quilted Speech',
        'roi': 'Speech Processing (STG)',
        'color_positive': 'Nonwords',
        'color_negative': 'Quilted',
        'modality': 'auditory'
    },
    'math_vs_theory_of_mind': {
        'label': 'Math > Theory of Mind',
        'roi': 'MD (Multiple Demand Network)',
        'color_positive': 'Math',
        'color_negative': 'ToM',
        'modality': 'auditory'
    },
    'english_vs_nonwords': {
        'label': 'English > Nonwords',
        'roi': 'Language Network',
        'color_positive': 'English Stories',
        'color_negative': 'Nonwords',
        'modality': 'auditory'
    }
}

# =============================================================================
# Parcel-to-Contrast Mapping (for fROI definition)
# =============================================================================

PARCEL_CONTRAST_MAP = {
    # Visual ROIs (Julian parcels)
    'ffa': 'faces_vs_objects',
    'ofa': 'faces_vs_objects',
    'sts': 'faces_vs_objects',  # Face-selective STS
    'ppa': 'scenes_vs_objects',
    'opa': 'scenes_vs_objects',
    'rsc': 'scenes_vs_objects',
    'eba': 'bodies_vs_objects',
    'loc': 'objects_vs_words',
    'vwfa': 'words_vs_objects',
    
    # Theory of Mind (ToM parcels)
    'tpj': 'false_belief_vs_false_photo',
    'mmpfc': 'false_belief_vs_false_photo',
    'vmpfc': 'false_belief_vs_false_photo',
    'dmpfc': 'false_belief_vs_false_photo',
    'pc': 'false_belief_vs_false_photo',
    
    # Language (language parcels)
    'ifg': 'english_vs_nonwords',
    'ifgorb': 'english_vs_nonwords',
    'mfg': 'english_vs_nonwords',
    'anttemp': 'english_vs_nonwords',
    'posttemp': 'english_vs_nonwords',
    'ag': 'english_vs_nonwords',
    
    # Speech
    'speech': 'nonwords_vs_quilted',
    
    # Multiple Demand (MD parcels) - use math contrast
    'antparietal': 'math_vs_theory_of_mind',
    'midparietal': 'math_vs_theory_of_mind',
    'postparietal': 'math_vs_theory_of_mind',
    'insula': 'math_vs_theory_of_mind',
    'supfrontal': 'math_vs_theory_of_mind',
    'midfrontal': 'math_vs_theory_of_mind',
    'medialfrontal': 'math_vs_theory_of_mind',
    'midfrontalorb': 'math_vs_theory_of_mind',
    'precentral_a_precg': 'math_vs_theory_of_mind',
    'precentral_b_ifgop': 'math_vs_theory_of_mind',
}

# Parcel categories
PARCEL_CATEGORIES = {
    'julian': ['ffa', 'ofa', 'sts', 'ppa', 'opa', 'rsc', 'eba', 'loc'],  # vwfa in separate category
    'language': ['ifg', 'ifgorb', 'mfg', 'anttemp', 'posttemp', 'ag'],
    'tom': ['tpj', 'mmpfc', 'vmpfc', 'dmpfc', 'pc'],
    'md': ['antparietal', 'midparietal', 'postparietal', 'insula', 'supfrontal', 'midfrontal', 'medialfrontal', 'precentral_a_precg', 'precentral_b_ifgop'],
    'speech': ['speech'],
    'vwfa': ['vwfa'],  # Left hemisphere only
}

# =============================================================================
# Path Configuration
# =============================================================================

def get_parcels_dir():
    """
    Get path to anatomical parcels directory.

    Resolution order (release, path-agnostic):
      1. ``MARVI_PARCELS_DIR`` environment variable, if set.
      2. Release-vendored ``Marvi_2025/data/PARCELS/`` (relative to this file).
      3. Dev-repo fallback ``…/src/aux/emfl_analysis-main/PARCELS`` (so the port runs
         before the 7.5M of parcel niftis are vendored — see README §8).

    Callers (drivers, tests) normally pass ``--parcels-dir`` explicitly and only fall
    back to this helper.

    Returns
    -------
    Path
        Path to PARCELS directory (may not exist; caller should check).
    """
    import os

    env = os.environ.get("MARVI_PARCELS_DIR")
    if env:
        return Path(env)

    # Release layout: this file is Marvi_2025/emfl/config.py → data/ is a sibling of emfl/.
    # The 6 used parcel categories (~6.8 MB) are vendored in the repo (see data/PROVENANCE.md);
    # this is the standalone source of truth for a fresh clone.
    dataset_dir = Path(__file__).resolve().parent.parent  # Marvi_2025/
    vendored = dataset_dir / 'data' / 'PARCELS'
    if vendored.exists():
        return vendored

    # RELEASE PORT NOTE: there is deliberately NO dev-repo fallback. The old fallback
    # (…/20251030_Marvi_2025…/PARCELS) silently "worked" only on the author's machine and
    # hid a broken standalone build (parcels present neither in git nor in the OSF cut). If
    # the vendored copy is missing, fail loudly rather than reach into an absolute dev path.
    raise FileNotFoundError(
        f"Anatomical parcels not found at {vendored}. They are vendored in the release "
        f"(Marvi_2025/data/PARCELS/); if you removed them, restore them or set "
        f"$MARVI_PARCELS_DIR to a PARCELS dir with the julian/language/tom/md/speech/vwfa "
        f"categories. See Marvi_2025/data/PROVENANCE.md.")


# =============================================================================
# fROI Parameters
# =============================================================================

# Default percentile for fROI selection (top N% of voxels)
DEFAULT_FROI_PERCENTILE = 10.0

# Minimum number of voxels for valid fROI
MIN_FROI_VOXELS = 10

# =============================================================================
# Output Space Configuration
# =============================================================================

# Available output spaces
VALID_SPACES = [
    'MNI152NLin2009cAsym',  # Standard MNI volumetric
    'T1w',                   # Native anatomical volumetric
    'fsaverage5',            # FreeSurfer template surface
    'fsnative'               # FreeSurfer native surface
]

# Default space for analysis (MNI space - parcels already in MNI, no warping needed)
DEFAULT_SPACE = 'MNI152NLin2009cAsym'

# =============================================================================
# Visualization Parameters
# =============================================================================

# Default p-value threshold for visualization
DEFAULT_P_THRESHOLD = 0.001

# Multiple comparison correction methods
CORRECTION_METHODS = ['uncorrected', 'FDR', 'bonferroni']

# Default correction method
DEFAULT_CORRECTION = 'uncorrected'

# Colormaps
SURFACE_COLORMAP = 'coolwarm'
VOLUME_COLORMAP = 'cold_hot'

# =============================================================================
# Helper Functions
# =============================================================================

def get_contrast_modality(contrast_name):
    """
    Get modality (visual/auditory) for a contrast.
    
    Parameters
    ----------
    contrast_name : str
        Contrast name
        
    Returns
    -------
    str
        'visual' or 'auditory'
    """
    if contrast_name in VISUAL_CONTRASTS:
        return 'visual'
    elif contrast_name in AUDITORY_CONTRASTS:
        return 'auditory'
    else:
        raise ValueError(f"Unknown contrast: {contrast_name}")


def get_runs_for_split(run_split='all'):
    """
    Get list of runs for specified split.
    
    Parameters
    ----------
    run_split : str
        'all', 'even', or 'odd'
        
    Returns
    -------
    list
        List of run identifiers
    """
    if run_split not in RUN_SPLITS:
        raise ValueError(f"Invalid run_split: {run_split}. Must be one of {list(RUN_SPLITS.keys())}")
    return RUN_SPLITS[run_split]


def is_surface_space(space):
    """
    Check if space is surface-based.
    
    Parameters
    ----------
    space : str
        Space identifier
        
    Returns
    -------
    bool
        True if surface space
    """
    return 'fs' in space.lower()

