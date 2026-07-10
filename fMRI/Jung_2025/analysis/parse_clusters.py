#!/usr/bin/env python3
"""Parse the 54-cluster (new54) model cluster-assignment JSON into a per-snippet fMRI
TR-timing CSV.

Faithful port of dev `src/44_parse_new54clusters_json.py`
(20251211_fMRI_movie_watching_spacetop @ 4066746). All six published brain-validation
maps are single discovered clusters from this one new54 partition (App. D "14 clusters"
is a typo for 54 — README §4); the dev 14-/21-/22-cluster + supercluster parsers are
NOT ported (docs/DESIGN.md §8 / index §7).

The JSON gives each cluster a list of ``video_ids``; the label is auto-derived from the
dominant video content and clip timing from the chunk index (each chunk = 2 s). Video
onsets are run-relative, read from sub-0001's ``task-alignvideo`` ``events.tsv`` under
``--raw-root`` (all subjects saw the identical stimulus order). Output columns match the
dev CSV exactly so the engine's regressor builder consumes it unchanged and the golden
master is bitwise.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

TR = 0.46  # seconds

SESSION_RUNS = {
    "ses-01": [f"run-{i:02d}" for i in range(1, 5)],
    "ses-02": [f"run-{i:02d}" for i in range(1, 5)],
    "ses-03": [f"run-{i:02d}" for i in range(1, 4)],
    "ses-04": [f"run-{i:02d}" for i in range(1, 3)],
}

CSV_COLUMNS = [
    'cluster_id', 'cluster_label', 'clip_filename', 'video_name',
    'clip_start', 'clip_end', 'clip_duration', 'session', 'run',
    'video_onset', 'video_duration', 'clip_onset', 'clip_duration_actual',
    'clip_onset_TR', 'clip_end_TR',
]


def load_video_events(raw_root, subject='sub-0001'):
    """Load run-relative video-presentation events for the reference subject.

    Returns {(session, run): DataFrame[onset, duration, video_name]}.
    """
    print(f"Loading video events from {subject}...")
    video_events = {}
    for session, runs in SESSION_RUNS.items():
        for run in runs:
            events_file = (
                Path(raw_root) / subject / session / "func"
                / f"{subject}_{session}_task-alignvideo_acq-mb8_{run}_events.tsv"
            )
            if not events_file.exists():
                print(f"  WARNING: events file not found: {events_file}")
                continue
            events = pd.read_csv(events_file, sep='\t')
            video_trials = events[events['trial_type'] == 'video'].copy()
            video_trials['video_name'] = video_trials['stim_file'].str.extract(
                r'content-(.+)\.mp4'
            )
            video_events[(session, run)] = video_trials[['onset', 'duration', 'video_name']]
    print(f"  Loaded events for {len(video_events)} session/run combinations")
    return video_events


def parse_video_id(video_id):
    """Extract (session, run, video_name, chunk_id) from a clip filename, or Nones."""
    basename = Path(video_id).name
    m = re.match(r'^(ses-\d+)_(run-\d+)_order-\d+_content-(.+)_chunk_(\d+)\.mp4$', basename)
    if not m:
        return None, None, None, None
    return m.group(1), m.group(2), m.group(3), int(m.group(4))


def derive_label(cluster_id, video_names):
    """Auto-derive a human-readable label from a cluster's video content."""
    counts = Counter(video_names)
    dominant = counts.most_common()
    if len(dominant) == 1:
        return dominant[0][0]
    total = sum(c for _, c in dominant)
    if dominant[0][1] / total > 0.8:
        return dominant[0][0]
    names = sorted(set(video_names))
    if len(names) <= 3:
        return '+'.join(names)
    return f"mixed_{cluster_id:02d}"


def _cluster_items(cluster_entry):
    """Yield (video_id, video_name, start_time, end_time) tuples for a cluster entry.

    Also returns the cluster label, derived from the dominant video content. Timing comes
    from the chunk index encoded in each ``video_id`` (each chunk = 2 s).
    """
    items = []
    failed = []
    for vid_id in cluster_entry['video_ids']:
        _, _, vname, chunk_id = parse_video_id(vid_id)
        if vname is None:
            failed.append({'cluster_id': cluster_entry['cluster_id'], 'video_id': vid_id,
                           'reason': 'filename parse error'})
            continue
        items.append((vid_id, vname, chunk_id * 2, chunk_id * 2 + 2))

    label = derive_label(cluster_entry['cluster_id'], [x[1] for x in items])
    return label, items, failed


def parse(json_file, raw_root):
    """Parse the new54 cluster JSON into the per-snippet TR-timing DataFrame."""
    json_path = Path(json_file)
    with open(json_path) as f:
        clusters = json.load(f)
    print(f"Loaded {len(clusters)} clusters from {json_path}")

    video_events = load_video_events(raw_root, 'sub-0001')

    records = []
    failed = []

    for cluster_entry in clusters:
        cluster_id = cluster_entry['cluster_id']
        cluster_label, items, item_failures = _cluster_items(cluster_entry)
        failed.extend(item_failures)
        print(f"  Cluster {cluster_id:02d} ({cluster_label}): {len(items)} clips")

        for video_id, video_name, start_time, end_time in items:
            clip_duration = end_time - start_time

            session, run, _, _ = parse_video_id(video_id)
            if session is None:
                failed.append({'cluster_id': cluster_id, 'video_id': video_id,
                               'reason': 'filename parse error'})
                continue

            if (session, run) not in video_events:
                failed.append({'cluster_id': cluster_id, 'video_id': video_id,
                               'reason': f'no events loaded for {session}/{run}'})
                continue

            events_df = video_events[(session, run)]
            match = events_df[events_df['video_name'] == video_name]
            if len(match) == 0:
                failed.append({'cluster_id': cluster_id, 'video_id': video_id,
                               'reason': f'"{video_name}" not found in {session}/{run} events'})
                continue

            video_onset = match.iloc[0]['onset']
            video_duration = match.iloc[0]['duration']

            actual_end = min(end_time, video_duration)
            actual_duration = actual_end - start_time
            if actual_duration <= 0:
                failed.append({'cluster_id': cluster_id, 'video_id': video_id,
                               'reason': 'clip start beyond video end'})
                continue

            clip_onset = video_onset + start_time
            clip_onset_TR = clip_onset / TR
            clip_end_TR = (video_onset + actual_end) / TR

            records.append({
                'cluster_id': cluster_id,
                'cluster_label': cluster_label,
                'clip_filename': video_id,
                'video_name': video_name,
                'clip_start': start_time,
                'clip_end': end_time,
                'clip_duration': clip_duration,
                'session': session,
                'run': run,
                'video_onset': video_onset,
                'video_duration': video_duration,
                'clip_onset': clip_onset,
                'clip_duration_actual': actual_duration,
                'clip_onset_TR': clip_onset_TR,
                'clip_end_TR': clip_end_TR,
            })

    df = pd.DataFrame(records, columns=CSV_COLUMNS)
    print(f"\nTotal rows: {len(df)} | clusters: {df['cluster_id'].nunique()} | failed: {len(failed)}")
    if failed:
        print(f"WARNING — {len(failed)} clips failed to map:")
        for x in failed[:10]:
            print(f"  [{x['cluster_id']:02d}] {x['video_id']}: {x['reason']}")
    return df


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--json-file', required=True, help='Vendored new54 model cluster-assignment JSON.')
    p.add_argument('--raw-root', required=True, help='OpenNeuro ds005256 BIDS root (for sub-0001 events).')
    p.add_argument('--out-csv', required=True, help='Destination cluster-assignments CSV.')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    df = parse(args.json_file, args.raw_root)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✓ Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
