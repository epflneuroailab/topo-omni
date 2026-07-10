"""core — shared utilities for the Omni-paper unified fMRI code release.

Minimal, version-robust helpers used across all three datasets (docs/DESIGN.md §4).
Installed editable per dataset env (`pip install -e core/`) and imported as `core`.

Must import cleanly under BOTH nilearn 0.10.4 (Pernet) and 0.12.1 (Marvi/Jung), so
keep top-level imports to numpy/scipy. nilearn-touching code (surface.py) imports
nilearn lazily/inside functions, never at module top level.

STATUS: scaffold — modules are stubs raising NotImplementedError.
"""

__all__ = ["spatial_stats", "surface", "paths", "froi_cv"]
__version__ = "0.0.0"
