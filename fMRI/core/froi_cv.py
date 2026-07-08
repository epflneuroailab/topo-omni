"""Cross-validated fROI extraction math — shared IFF it factors cleanly (docs/DESIGN.md §4).

Candidate consolidation of the cross-validated functional-ROI logic shared by Pernet
(`cv_*` lineage) and Marvi (`emfl.roi.*`): split-half define/measure, so a region is
defined on one partition and its response read out on the held-out partition.

⚠ CONDITIONAL: only lift here if Pernet's and Marvi's CV math factor cleanly. If they
diverge too much, leave each LOCAL — do not force it. Decision is made DURING the
Marvi port (docs/DESIGN.md §9 step 3, §10), once both concrete implementations are in hand.
Pure numpy/scipy -> version-robust if it does land here.

STATUS: scaffold — placeholder; may stay empty and be deleted if the factoring
doesn't hold.
"""
from __future__ import annotations

import numpy as np


def cross_validated_froi_response(*args, **kwargs):
    """Define fROI on one fold, read out response on the held-out fold.

    TODO(decide during Marvi port): does this factor out of Pernet cv_* + Marvi
    emfl.roi.*? If not, delete this module and keep both local (docs/DESIGN.md §4).
    """
    raise NotImplementedError("core.froi_cv — scaffold; factoring TBD during Marvi port (docs/DESIGN.md §4)")
