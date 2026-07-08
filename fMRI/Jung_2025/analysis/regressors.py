#!/usr/bin/env python3
"""Regressor construction for the Jung cluster-vs-others GLM.

Faithful port of the cluster path of dev `src/create_regressors_unified.py`
(20251211_fMRI_movie_watching_spacetop @ 4066746). The video, cluster-vs-rating, and
random-baseline regressor families are NOT ported — they are not in the paper
(docs/DESIGN.md §8 / index §7). The kept functions (HRF, convolution, run-info, cluster
regressors) are byte-faithful so the golden masters reproduce the published maps.

Design (per subject, per hemisphere): the GLM design matrix is
`[target, others, *24_confounds]`, contrast = `target - others`. `target` covers the
TRs of the target cluster's video snippets; `others` covers all OTHER video TRs;
emotion-rating TRs are left unmodeled as the implicit baseline. Both regressors are
HRF-convolved and z-scored.
"""

import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path

# Constants
TR = 0.46  # seconds
HRF_PEAK = 5.0  # seconds
HRF_UNDERSHOOT = 15.0  # seconds


def create_canonical_hrf(tr=TR, duration=32):
    """Create canonical (double-gamma, SPM-style) HRF sampled at TR intervals."""
    from scipy.stats import gamma as gamma_dist

    t = np.arange(0, duration, tr)

    # Canonical HRF: peak gamma(6,1) minus undershoot gamma(16,1)/6
    hrf = (
        gamma_dist.pdf(t, 6, scale=1) -
        gamma_dist.pdf(t, 16, scale=1) / 6
    )

    # Normalize to peak = 1
    if np.max(hrf) > 0:
        hrf = hrf / np.max(hrf)

    return hrf


def convolve_with_hrf(regressor, tr=TR):
    """Convolve a binary regressor with the canonical HRF, then z-score."""
    hrf = create_canonical_hrf(tr=tr)
    convolved = signal.fftconvolve(regressor, hrf, mode='same')

    # Z-score
    if np.std(convolved) > 0:
        convolved = (convolved - np.mean(convolved)) / np.std(convolved)

    return convolved


def load_run_info(subject_id, bold_dir):
    """Load run TR boundaries for a subject: dict {(session, run): (start_tr, end_tr)}.

    Boundaries come from the number of darrays (= TRs) in each fsaverage6 hemi-L GIFTI,
    concatenated in sorted (session, run) order.
    """
    import nibabel as nib

    run_info = {}
    cumulative_tr = 0

    bold_files = sorted(Path(bold_dir).glob(
        f'{subject_id}/ses-*/func/*_hemi-L_space-fsaverage6_bold.func.gii'
    ))

    if not bold_files:
        raise FileNotFoundError(f"No BOLD files found for {subject_id} in {bold_dir}")

    for bold_file in bold_files:
        parts = bold_file.stem.split('_')
        session = [p for p in parts if p.startswith('ses-')][0]
        run = [p for p in parts if p.startswith('run-')][0]

        bold_data = nib.load(str(bold_file))
        n_trs = len(bold_data.darrays)

        run_info[(session, run)] = (cumulative_tr, cumulative_tr + n_trs)
        cumulative_tr += n_trs

    return run_info


def create_cluster_regressors(cluster_id, cluster_df, events_df, run_info, return_binary=False):
    """Regressors for the CLUSTER contrast: target_cluster vs. all_other_clusters.

    Coverage: all VIDEO TRs (rating periods excluded / unmodeled). Every covered TR is
    in exactly one of {target, others}; no overlap. `others` marks ALL other videos'
    TRs (not only cluster videos), so an exclusive single-video cluster reproduces the
    video contrast.

    Returns HRF-convolved, z-scored (target, others), or the raw binary indicators if
    `return_binary=True`.
    """
    # Total TRs
    n_trs = max(end for _, end in run_info.values())

    # Binary indicators
    target_binary = np.zeros(n_trs)
    others_binary = np.zeros(n_trs)

    # Snippets (TR ranges) in the target cluster
    target_snippets = cluster_df[cluster_df['cluster_id'] == cluster_id]
    target_videos = target_snippets['video_name'].unique()

    print(f"  Cluster {cluster_id}: {len(target_snippets)} snippets from {len(target_videos)} videos = {set(target_videos)}")

    # Mark target cluster snippet TRs
    for _, snippet in target_snippets.iterrows():
        session = snippet['session']
        run = snippet['run']
        video_name = snippet['video_name']

        if (session, run) not in run_info:
            continue

        tr_start_global, tr_end_global = run_info[(session, run)]

        # Look up this video's run-relative onset in events_df
        video_event = events_df[
            (events_df['session'] == session) &
            (events_df['run'] == run) &
            (events_df['video_name'] == video_name)
        ]

        if len(video_event) == 0:
            continue

        video_onset_seconds = video_event.iloc[0]['onset_seconds']  # run-relative
        video_duration_seconds = video_event.iloc[0]['duration_seconds']

        # Snippet position within the video
        clip_start = snippet['clip_start']
        clip_end = snippet['clip_end']

        # Clip snippet boundaries to video duration (prevents rounding issues)
        clip_start = max(0, min(clip_start, video_duration_seconds))
        clip_end = max(clip_start, min(clip_end, video_duration_seconds))

        snippet_start_seconds = video_onset_seconds + clip_start
        snippet_end_seconds = video_onset_seconds + clip_end

        snippet_tr_start = int(snippet_start_seconds / TR)
        snippet_tr_end = int(snippet_end_seconds / TR)

        tr_start = tr_start_global + snippet_tr_start
        tr_end = tr_start_global + snippet_tr_end
        tr_start = max(0, min(tr_start, n_trs))
        tr_end = max(0, min(tr_end, n_trs))

        target_binary[tr_start:tr_end] = 1

    # Mark ALL OTHER VIDEO TRs as "others" (not just cluster videos)
    for _, event in events_df.iterrows():
        session = event['session']
        run = event['run']
        video_name = event['video_name']

        # Skip if this is a target video
        if video_name in target_videos:
            continue

        if (session, run) not in run_info:
            continue

        tr_start_global, tr_end_global = run_info[(session, run)]

        event_tr_start = int(event['onset_seconds'] / TR)
        event_tr_end = int((event['onset_seconds'] + event['duration_seconds']) / TR)

        tr_start = tr_start_global + event_tr_start
        tr_end = tr_start_global + event_tr_end
        tr_start = max(0, min(tr_start, n_trs))
        tr_end = max(0, min(tr_end, n_trs))

        others_binary[tr_start:tr_end] = 1

    # Validate coverage
    overlap = np.sum(target_binary * others_binary)
    if overlap > 0:
        raise ValueError(f"Overlap detected: {overlap} TRs in both target and others!")

    total_video_trs = np.sum(target_binary) + np.sum(others_binary)
    print(f"  Cluster coverage: {total_video_trs} TRs ({target_binary.sum():.0f} target + {others_binary.sum():.0f} others)")

    if return_binary:
        return target_binary, others_binary

    # HRF convolution
    target = convolve_with_hrf(target_binary)
    others = convolve_with_hrf(others_binary)

    return target, others


if __name__ == "__main__":
    print("Jung cluster regressor module loaded successfully")
    print(f"TR = {TR}s, HRF peak = {HRF_PEAK}s")
