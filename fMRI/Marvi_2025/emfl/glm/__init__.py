"""
GLM Analysis Modules
===================

First-level General Linear Model analysis.

Port note (release): the dev repo's group-level GLM (``group_level.EFMLOCGroupLevelGLM``)
is deny-listed (group-level fsaverage5 lineage — docs/DESIGN.md §7/§8) and NOT vendored, so its
import is dropped here. Only the individual-subject first-level engine is shipped.
"""

from emfl.glm.first_level import EFMLOCFirstLevelGLM

__all__ = [
    'EFMLOCFirstLevelGLM',
]

