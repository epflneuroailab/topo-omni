"""
EMFL Analysis Package
=====================

Python package for analyzing fMRI data from the Marvi et al. (2025) 
Efficient Multi-Functional Localizer (EMFL).

This package provides a complete pipeline for:
- Preprocessing fMRI data with fMRIprep
- First-level and group-level GLM analysis
- Functional ROI (fROI) definition and extraction
- Cross-validation and visualization

Modules
-------
io : Input/output utilities
    BIDS file handling and event parsing
glm : GLM analysis
    First-level and group-level statistical modeling
roi : fROI analysis
    Definition, extraction, and cross-validation
visualization : Plotting and visualization
    Interactive and static brain visualizations
utils : Common utilities
    Helper functions and wrappers

Examples
--------
>>> from emfl.glm import EFMLOCFirstLevelGLM
>>> from emfl.config import ALL_SUBJECTS, DEFAULT_SMOOTHING
>>> 
>>> analyzer = EFMLOCFirstLevelGLM(
...     derivatives_dir='/path/to/derivatives',
...     subject_id='sub-kaneff01',
...     smoothing_fwhm=DEFAULT_SMOOTHING
... )
"""

__version__ = '1.0.0'
__author__ = 'Analysis Pipeline for Marvi et al. (2025)'

# Note: Specific imports will be added as modules are created
# Example: from emfl.glm import EFMLOCFirstLevelGLM

__all__ = [
    '__version__',
]

