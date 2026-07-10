#!/usr/bin/env python3
"""Unified fsaverage6 GLM engine for the Jung cluster-vs-others contrast.

Faithful port of the **cluster** path of dev `src/glm_unified.py`
(20251211_fMRI_movie_watching_spacetop @ 4066746), plus the confound and
canonical-subject constants it imported (dev `confounds_standard.py`,
`canonical_subjects.py`). The video / cluster-vs-rating / random-baseline paths are NOT
ported (not in the paper — docs/DESIGN.md §8 / index §7).

This is a **dataset-specific engine** and stays local (docs/DESIGN.md §4). Unlike Pernet's
`SecondLevelModel`, it is pure numpy / scipy / nibabel / pandas — **no nilearn** — so it
is expected to reproduce the on-disk group maps to near machine precision under the
pinned Jung env (nilearn 0.12.1 / numpy 2.2.5); the golden-master tolerance is
calibrated then frozen (docs/DESIGN.md §6; Jung_2025/tests).

Pipeline (`run_full_analysis`): for each canonical subject, build the design matrix
`[target, others, *24_confounds]`, fit OLS per hemisphere, compute the `target - others`
contrast t-map, save per-subject t-maps; then group one-sample t-test across subjects,
one-tailed FDR q<0.05, save group `tstat / pval / mean` GIFTIs + summary.json.

**The n=78 drop lives here** (docs/DESIGN.md §7): `load_confounds` raises when a run is missing
any of the 24 named confounds; `run_subject_analysis` catches every exception and returns
`(None, None)`, dropping the subject. Exactly 5 of the 83 canonical subjects drop
(0035/0044/0061/0084/0131) → n=78, df=77. This IS the published behavior — kept, not
fixed (pinned by a characterization test).
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats
from pathlib import Path

# Import the cluster regressor builder (same analysis/ dir; robust under path-loading).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regressors import create_cluster_regressors, load_run_info  # noqa: E402

# --- Constants (dev confounds_standard.py + canonical_subjects.py, verbatim) ---
TR = 0.46

# The 24 standard confounds: Friston-24 motion + 5 aCompCor + 3 tCompCor + 4 cosine.
STANDARD_CONFOUNDS = [
    'trans_x', 'trans_y', 'trans_z',
    'rot_x', 'rot_y', 'rot_z',
    'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
    'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
    'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04',
    't_comp_cor_00', 't_comp_cor_01', 't_comp_cor_02',
    'cosine00', 'cosine01', 'cosine02', 'cosine03',
]
N_CONFOUNDS = 24

# The fixed 83 canonical subjects (dev canonical_subjects.py). 5 drop in load_confounds.
CANONICAL_SUBJECTS = [
    'sub-0001', 'sub-0002', 'sub-0003', 'sub-0004', 'sub-0005', 'sub-0006', 'sub-0008', 'sub-0010',
    'sub-0013', 'sub-0014', 'sub-0016', 'sub-0018', 'sub-0019', 'sub-0020', 'sub-0021', 'sub-0025',
    'sub-0026', 'sub-0029', 'sub-0031', 'sub-0032', 'sub-0033', 'sub-0034', 'sub-0035', 'sub-0037',
    'sub-0038', 'sub-0040', 'sub-0043', 'sub-0044', 'sub-0046', 'sub-0050', 'sub-0051', 'sub-0052',
    'sub-0053', 'sub-0055', 'sub-0058', 'sub-0059', 'sub-0060', 'sub-0061', 'sub-0062', 'sub-0064',
    'sub-0065', 'sub-0066', 'sub-0069', 'sub-0070', 'sub-0075', 'sub-0076', 'sub-0077', 'sub-0078',
    'sub-0079', 'sub-0080', 'sub-0083', 'sub-0084', 'sub-0086', 'sub-0087', 'sub-0088', 'sub-0089',
    'sub-0090', 'sub-0092', 'sub-0093', 'sub-0094', 'sub-0095', 'sub-0098', 'sub-0099', 'sub-0100',
    'sub-0101', 'sub-0102', 'sub-0104', 'sub-0105', 'sub-0106', 'sub-0107', 'sub-0109', 'sub-0111',
    'sub-0112', 'sub-0115', 'sub-0116', 'sub-0122', 'sub-0126', 'sub-0127', 'sub-0129', 'sub-0130',
    'sub-0131', 'sub-0132', 'sub-0133',
]
N_SUBJECTS = len(CANONICAL_SUBJECTS)


def load_confounds(subject_id, confounds_dir):
    """Load the standard 24 confounds for a subject, concatenated across runs, z-scored.

    Raises ValueError if any of the 24 named confounds is absent from any run — this is
    the mechanism that drops 5 subjects to yield the published n=78 (docs/DESIGN.md §7).
    """
    confound_files = sorted(Path(confounds_dir).glob(
        f'{subject_id}/ses-*/func/*_desc-confounds_timeseries.tsv'
    ))

    if not confound_files:
        raise FileNotFoundError(f"No confound files found for {subject_id}")

    all_confounds = []

    for conf_file in confound_files:
        df = pd.read_csv(conf_file, sep='\t')

        selected = []
        for confound in STANDARD_CONFOUNDS:
            if confound in df.columns:
                selected.append(df[confound].values)
            else:
                # Handle naming variations (e.g. a_comp_cor_00 vs aCompCor00)
                found = False
                for col in df.columns:
                    if confound.replace('_', '').lower() in col.replace('_', '').lower():
                        selected.append(df[col].values)
                        found = True
                        break
                if not found:
                    raise ValueError(f"Confound {confound} not found in {conf_file}")

        all_confounds.append(np.column_stack(selected))

    confounds = np.vstack(all_confounds)

    if confounds.shape[1] != N_CONFOUNDS:
        raise ValueError(f"Expected {N_CONFOUNDS} confounds, got {confounds.shape[1]}")

    # Fill NaNs with 0, then z-score each confound column
    confounds = np.nan_to_num(confounds, nan=0.0)
    for i in range(confounds.shape[1]):
        if np.std(confounds[:, i]) > 0:
            confounds[:, i] = (confounds[:, i] - np.mean(confounds[:, i])) / np.std(confounds[:, i])

    return confounds


def load_bold(subject_id, hemi, bold_dir):
    """Load fsaverage6 BOLD for one hemisphere: (n_trs, n_vertices), runs concatenated."""
    bold_files = sorted(Path(bold_dir).glob(
        f'{subject_id}/ses-*/func/*_hemi-{hemi}_space-fsaverage6_bold.func.gii'
    ))

    if not bold_files:
        raise FileNotFoundError(f"No BOLD files found for {subject_id} hemi-{hemi}")

    all_bold = []
    for bold_file in bold_files:
        gii = nib.load(str(bold_file))
        timepoints = [darray.data for darray in gii.darrays]
        run_data = np.vstack(timepoints)
        all_bold.append(run_data)

    return np.vstack(all_bold)


def fit_glm(bold, design_matrix):
    """OLS fit: betas = (X'X)^-1 X'Y; return (betas, residuals)."""
    XtX_inv = np.linalg.inv(design_matrix.T @ design_matrix)
    betas = XtX_inv @ design_matrix.T @ bold
    predicted = design_matrix @ betas
    residuals = bold - predicted
    return betas, residuals


def compute_contrast_tstat(betas, residuals, design_matrix, contrast_vector):
    """t-statistics for a contrast across vertices."""
    n_trs, n_regressors = design_matrix.shape

    contrast_estimate = contrast_vector @ betas  # (n_vertices,)

    df = n_trs - n_regressors
    residual_var = np.sum(residuals ** 2, axis=0) / df  # (n_vertices,)

    XtX_inv = np.linalg.inv(design_matrix.T @ design_matrix)
    contrast_var = contrast_vector @ XtX_inv @ contrast_vector  # scalar

    se = np.sqrt(contrast_var * residual_var)
    t_stats = contrast_estimate / (se + 1e-10)
    return t_stats


def compute_group_statistics(subject_tstats):
    """Group one-sample t-test (vs 0) over subject contrast t-maps; return (t, mean)."""
    data = np.array(subject_tstats)  # (n_subjects, n_vertices)
    group_t, _ = stats.ttest_1samp(data, 0, axis=0, nan_policy='omit')
    group_mean = np.nanmean(data, axis=0)
    return group_t, group_mean


def compute_fdr_threshold_onetailed(t_stats, q=0.05, df=82):
    """One-tailed (positive) BH-FDR threshold; return (t_threshold, n_significant)."""
    pos_t = t_stats[t_stats > 0]
    if len(pos_t) == 0:
        return np.inf, 0

    p_values = stats.t.sf(pos_t, df=df)
    sorted_p = np.sort(p_values)
    n_tests = len(sorted_p)

    significant_mask = sorted_p <= (np.arange(1, n_tests + 1) / n_tests) * q

    if np.any(significant_mask):
        fdr_p_threshold = sorted_p[significant_mask][-1]
        fdr_t_threshold = stats.t.ppf(1 - fdr_p_threshold, df=df)
        n_sig = np.sum(pos_t >= fdr_t_threshold)
        return fdr_t_threshold, n_sig
    return np.inf, 0


def save_gifti(data, output_path, hemi):
    """Save a (n_vertices,) array as a float32 GIFTI surface file."""
    darray = nib.gifti.GiftiDataArray(
        data=data.astype(np.float32),
        intent='NIFTI_INTENT_NONE',
        datatype='NIFTI_TYPE_FLOAT32',
    )
    img = nib.gifti.GiftiImage(darrays=[darray])
    nib.save(img, output_path)


def load_subject_events(subject_id, bids_dir):
    """Load task-alignvideo events for a subject → DataFrame with run-relative timing.

    Columns: [video_name, onset_seconds, duration_seconds, session, run].
    """
    all_events = []

    event_files = sorted(Path(bids_dir).glob(
        f'{subject_id}/ses-*/func/*_task-alignvideo_*_events.tsv'
    ))

    for event_file in event_files:
        parts = event_file.stem.split('_')
        session = [p for p in parts if p.startswith('ses-')][0]
        run = [p for p in parts if p.startswith('run-')][0]

        events = pd.read_csv(event_file, sep='\t')

        video_events = events[
            (events['trial_type'] == 'video') &
            (events['stim_file'] != 'n/a')
        ].copy()

        if len(video_events) == 0:
            continue

        video_events['video_name'] = video_events['stim_file'].str.extract(
            r'content-(.+)\.mp4'
        )[0]

        video_events['session'] = session
        video_events['run'] = run
        video_events['onset_seconds'] = video_events['onset']
        video_events['duration_seconds'] = video_events['duration']

        all_events.append(video_events[['video_name', 'onset_seconds', 'duration_seconds',
                                        'session', 'run']])

    if len(all_events) == 0:
        raise ValueError(f"No events found for {subject_id}")

    return pd.concat(all_events, ignore_index=True)


def run_subject_analysis(subject_id, target_id, bold_dir, confounds_dir, cluster_df, bids_dir):
    """Run the cluster GLM for one subject → (t_lh, t_rh), or (None, None) on any error.

    The blanket except is the published behavior: a subject whose confounds fail to load
    (missing named columns) is silently dropped (docs/DESIGN.md §7).
    """
    print(f"\nProcessing {subject_id}...")
    sys.stdout.flush()

    try:
        events_df = load_subject_events(subject_id, bids_dir)
        run_info = load_run_info(subject_id, bold_dir)

        target, contrast = create_cluster_regressors(target_id, cluster_df, events_df, run_info)

        confounds = load_confounds(subject_id, confounds_dir)

        design_matrix = np.column_stack([target, contrast, confounds])

        bold_lh = load_bold(subject_id, 'L', bold_dir)
        bold_rh = load_bold(subject_id, 'R', bold_dir)

        betas_lh, residuals_lh = fit_glm(bold_lh, design_matrix)
        betas_rh, residuals_rh = fit_glm(bold_rh, design_matrix)

        # Contrast: target - others (confounds contribute 0)
        contrast_vector = np.zeros(design_matrix.shape[1])
        contrast_vector[0] = 1
        contrast_vector[1] = -1

        t_lh = compute_contrast_tstat(betas_lh, residuals_lh, design_matrix, contrast_vector)
        t_rh = compute_contrast_tstat(betas_rh, residuals_rh, design_matrix, contrast_vector)

        print(f"  LH: t-range [{t_lh.min():.2f}, {t_lh.max():.2f}]")
        print(f"  RH: t-range [{t_rh.min():.2f}, {t_rh.max():.2f}]")
        sys.stdout.flush()

        return t_lh, t_rh

    except Exception as e:
        print(f"  ERROR: {e}")
        sys.stdout.flush()
        return None, None


def load_subject_tmaps(subject_maps_root, target_id, subjects=None):
    """Reuse mode: load per-subject t-maps written by an earlier (heavy) GLM run.

    This is the Tier-1 precomputed-cut path (docs/DESIGN.md §5.1-C). The shipped
    `<subject_maps_root>/subject_level/cluster-XX/sub-*/sub-*_hemi-{L,R}_tstat.func.gii`
    maps replace the ~1.5 TB fsaverage6 BOLD re-fit: we load them and fall straight into the
    deterministic group-level t-test/FDR. Subjects are visited in CANONICAL_SUBJECTS order
    (not directory order) so the group-stats summation order — and thus the group maps —
    reproduces the heavy run bit-for-bit. n=78 / df=77 falls out naturally: the 5
    confound-dropped subjects (0035/0044/0061/0084/0131) simply have no shipped dir.

    ``subject_maps_root`` is the dir that CONTAINS ``subject_level/`` — under the precomputed
    path this is the read-only cut (derivatives-root/<family subdir>), which is separate from
    the results-root where group maps + plots are written.

    Returns (results_lh, results_rh, successful_subjects), matching the compute path.
    """
    subject_root = Path(subject_maps_root) / 'subject_level' / f'cluster-{target_id:02d}'
    if not subject_root.is_dir():
        raise FileNotFoundError(
            f"reuse_subject_maps: no shipped subject-level t-maps at {subject_root}. The "
            f"precomputed cut must include cluster_contrasts_new54clusters/subject_level/ "
            f"(docs/DESIGN.md §5.1-C), or run --input-source raw to recompute from the fsaverage6 BOLD."
        )
    order = subjects if subjects is not None else CANONICAL_SUBJECTS
    results_lh, results_rh, successful_subjects = [], [], []
    for subject_id in order:
        lh = subject_root / subject_id / f'{subject_id}_hemi-L_tstat.func.gii'
        rh = subject_root / subject_id / f'{subject_id}_hemi-R_tstat.func.gii'
        if not (lh.exists() and rh.exists()):
            continue  # dropped subject (or not shipped) — expected for the 5 confound drops
        results_lh.append(nib.load(str(lh)).darrays[0].data)
        results_rh.append(nib.load(str(rh)).darrays[0].data)
        successful_subjects.append(subject_id)
    return results_lh, results_rh, successful_subjects


def run_full_analysis(target_id, output_dir, bold_dir, confounds_dir, bids_dir,
                      cluster_file, subjects=None, reuse_subject_maps=False,
                      subject_maps_root=None):
    """Run the complete cluster contrast: subject-level GLMs + group-level t-test/FDR.

    Writes per-subject t-maps under `subject_level/cluster-XX/` and group
    `tstat/pval/mean` GIFTIs + summary.json under `group_level/`. Returns True on success.

    ``reuse_subject_maps=True`` (the precomputed-cut / render-only path, docs/DESIGN.md §5.1-C):
    skip the per-subject GLM entirely and load the shipped `subject_level/` t-maps instead
    of reading the excluded ~1.5 TB fsaverage6 BOLD + confounds + raw events. The group
    stage is identical either way, so the group maps reproduce bit-for-bit.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("UNIFIED GLM ANALYSIS (cluster vs. all others)")
    print("=" * 80)
    print(f"Target cluster: {target_id}")
    print(f"Output: {output_dir}")
    print(f"Confounds: {N_CONFOUNDS} standard")
    print("=" * 80)
    sys.stdout.flush()

    if reuse_subject_maps:
        maps_root = subject_maps_root if subject_maps_root is not None else output_dir
        print(f"\nReuse mode: loading shipped subject-level t-maps from {maps_root} "
              f"(no BOLD re-fit) ...")
        sys.stdout.flush()
        results_lh, results_rh, successful_subjects = load_subject_tmaps(
            maps_root, target_id, subjects
        )
    else:
        cluster_df = pd.read_csv(cluster_file)

        if subjects is None:
            subjects = CANONICAL_SUBJECTS

        print(f"\nAnalyzing {len(subjects)} subjects...")
        sys.stdout.flush()

        results_lh = []
        results_rh = []
        successful_subjects = []

        for subject_id in subjects:
            t_lh, t_rh = run_subject_analysis(
                subject_id, target_id, bold_dir, confounds_dir, cluster_df, bids_dir
            )

            if t_lh is not None and t_rh is not None:
                results_lh.append(t_lh)
                results_rh.append(t_rh)
                successful_subjects.append(subject_id)

                subject_subdir = f'cluster-{target_id:02d}'
                subject_dir = output_dir / 'subject_level' / subject_subdir / subject_id
                subject_dir.mkdir(parents=True, exist_ok=True)
                save_gifti(t_lh, subject_dir / f'{subject_id}_hemi-L_tstat.func.gii', 'L')
                save_gifti(t_rh, subject_dir / f'{subject_id}_hemi-R_tstat.func.gii', 'R')

    print(f"\n{len(successful_subjects)} subjects included in the group analysis")
    sys.stdout.flush()

    if len(successful_subjects) < 3:
        print("ERROR: Not enough subjects for group analysis")
        return False

    print("\n" + "=" * 80)
    print("GROUP-LEVEL ANALYSIS")
    print("=" * 80)
    sys.stdout.flush()

    group_t_lh, group_mean_lh = compute_group_statistics(results_lh)
    group_t_rh, group_mean_rh = compute_group_statistics(results_rh)

    df = len(successful_subjects) - 1

    print(f"Group t-stats (df={df}):")
    print(f"  LH: range [{group_t_lh.min():.2f}, {group_t_lh.max():.2f}]")
    print(f"  RH: range [{group_t_rh.min():.2f}, {group_t_rh.max():.2f}]")
    sys.stdout.flush()

    # One-tailed p-values (upper tail: positive activation)
    group_p_lh = stats.t.sf(group_t_lh, df=df)
    group_p_rh = stats.t.sf(group_t_rh, df=df)

    fdr_t_lh, n_sig_lh = compute_fdr_threshold_onetailed(group_t_lh, q=0.05, df=df)
    fdr_t_rh, n_sig_rh = compute_fdr_threshold_onetailed(group_t_rh, q=0.05, df=df)

    print(f"\nFDR q<0.05 (one-tailed):")
    print(f"  LH: threshold t={fdr_t_lh:.2f}, n_sig={n_sig_lh}")
    print(f"  RH: threshold t={fdr_t_rh:.2f}, n_sig={n_sig_rh}")
    sys.stdout.flush()

    group_dir = output_dir / 'group_level'
    group_dir.mkdir(parents=True, exist_ok=True)

    prefix = f'group_cluster-{target_id:02d}_space-fsaverage6'

    save_gifti(group_t_lh, group_dir / f'{prefix}_hemi-L_tstat.func.gii', 'L')
    save_gifti(group_t_rh, group_dir / f'{prefix}_hemi-R_tstat.func.gii', 'R')
    save_gifti(group_p_lh, group_dir / f'{prefix}_hemi-L_pval.func.gii', 'L')
    save_gifti(group_p_rh, group_dir / f'{prefix}_hemi-R_pval.func.gii', 'R')
    save_gifti(group_mean_lh, group_dir / f'{prefix}_hemi-L_mean.func.gii', 'L')
    save_gifti(group_mean_rh, group_dir / f'{prefix}_hemi-R_mean.func.gii', 'R')

    summary = {
        'regressor_type': 'cluster',
        'target_id': str(target_id),
        'n_subjects': len(successful_subjects),
        'df': df,
        'max_t_lh': float(group_t_lh.max()),
        'max_t_rh': float(group_t_rh.max()),
        'fdr_t_lh': float(fdr_t_lh),
        'fdr_t_rh': float(fdr_t_rh),
        'n_sig_lh': int(n_sig_lh),
        'n_sig_rh': int(n_sig_rh),
        'n_sig_total': int(n_sig_lh + n_sig_rh),
    }
    with open(group_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 80)
    sys.stdout.flush()

    return True


if __name__ == "__main__":
    print("Jung GLM engine loaded successfully")
    print(f"Standard confounds: {N_CONFOUNDS}; canonical subjects: {N_SUBJECTS}")
