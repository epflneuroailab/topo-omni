"""Pernet 2015 Stage-0 preprocessing package (FSL + Nilearn first-level GLM).

Faithful port of the dev pipeline (20241003_pernet_2015 @ f842b1a). Stage 0 is
env-pinned (nilearn 0.10.4) and FSL-dependent, so it is NOT golden-mastered
(docs/DESIGN.md \u00a72.5/\u00a76): faithful port + raw-dispatch smoke test. Heavy deps
(nibabel/nilearn/FSL) are imported inside the modules that use them, so importing
this package is only as heavy as the submodule you touch.
"""
