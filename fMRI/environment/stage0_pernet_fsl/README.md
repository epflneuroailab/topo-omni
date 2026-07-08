# Stage-0 toolchain — Pernet (FSL)

FSL 6.0.7 (+ Nilearn) volumetric preprocessing. Runs in a temp dir; **preprocessed
BOLD is never written to disk** — smoothed BOLD is handed to the Nilearn GLM in
memory, so Pernet's precomputed cut is contrast-level, not derivatives (docs/DESIGN.md §1).

STATUS: scaffold — pin container/version + provenance spot-check list during the
Pernet port (docs/DESIGN.md §9 step 2).
