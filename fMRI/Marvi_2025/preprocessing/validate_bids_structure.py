"""Stage 0 (step 1) — validate the BIDS structure of the Marvi 2025 EMFL dataset.

Scans the raw BIDS directory and reports, per subject: tasks/runs, event-file presence,
anatomical (T1w) presence, and which subjects have complete EMFL (``effloc``, 5 runs) data —
the completeness check that gates fMRIPrep. Only used by the ``--input-source raw`` path; the
default ``precomputed`` reproduction never runs this (docs/DESIGN.md §2.5/§6).

Parameterized (no baked-in dataset path):

    python validate_bids_structure.py --bids-dir <ds006179 BIDS root> [--output-csv summary.csv]
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

# BIDS data directory — set from --bids-dir in main(); functions below read it as a module global.
BIDS_DIR = None

def scan_bids_directory():
    """
    Scan BIDS directory and collect information about subjects, tasks, and runs.
    
    Returns:
        dict: Nested dictionary with subject -> task -> runs information
    """
    data_summary = defaultdict(lambda: defaultdict(lambda: {'runs': [], 'events': [], 'bold': []}))
    
    # Get all subject directories
    subject_dirs = sorted([d for d in BIDS_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    print(f"Found {len(subject_dirs)} subject directories")
    print(f"Subject IDs: {[d.name for d in subject_dirs]}\n")
    
    for sub_dir in subject_dirs:
        subject_id = sub_dir.name
        func_dir = sub_dir / 'func'
        
        if not func_dir.exists():
            print(f"WARNING: No func directory for {subject_id}")
            continue
            
        # Get all files in func directory
        func_files = list(func_dir.glob('*.nii.gz')) + list(func_dir.glob('*.json')) + list(func_dir.glob('*.tsv'))
        
        for f in func_files:
            # Parse filename for task and run information
            # Expected format: sub-<label>_task-<label>_run-<label>_<suffix>
            parts = f.stem.replace('.nii', '').split('_')
            
            task = None
            run = None
            suffix = None
            
            for part in parts:
                if part.startswith('task-'):
                    task = part.split('-')[1]
                elif part.startswith('run-'):
                    run = part.split('-')[1]
            
            # Get suffix (last part, e.g., 'bold', 'events')
            if f.suffix == '.gz':
                suffix = 'bold'
            elif f.suffix == '.tsv':
                suffix = 'events'
            elif f.suffix == '.json':
                suffix = 'json'
            
            if task and suffix:
                if suffix == 'bold' and run:
                    data_summary[subject_id][task]['bold'].append(run)
                elif suffix == 'events' and run:
                    data_summary[subject_id][task]['events'].append(run)
                elif suffix == 'events' and not run:
                    # Some event files might not have run labels
                    data_summary[subject_id][task]['events'].append('001')
    
    return dict(data_summary)

def check_completeness(data_summary):
    """
    Check which subjects have complete EMFL (effloc) data.
    
    According to the paper, EMFL consists of 5 runs.
    
    Args:
        data_summary: Dictionary from scan_bids_directory()
        
    Returns:
        dict: Summary of data completeness per subject
    """
    completeness = {}
    
    for subject_id, tasks in data_summary.items():
        completeness[subject_id] = {}
        
        for task, files in tasks.items():
            n_bold = len(files['bold'])
            n_events = len(files['events'])
            
            # Check if both BOLD and events are present
            has_both = n_bold > 0 and n_events > 0
            
            # For effloc, we expect 5 runs according to the paper
            is_complete_effloc = (task == 'effloc' and n_bold == 5 and n_events == 5)
            
            completeness[subject_id][task] = {
                'n_bold_runs': n_bold,
                'n_event_files': n_events,
                'has_both': has_both,
                'is_complete': is_complete_effloc if task == 'effloc' else has_both
            }
    
    return completeness

def generate_summary_report(data_summary, completeness):
    """
    Generate a comprehensive summary report.
    
    Args:
        data_summary: Dictionary from scan_bids_directory()
        completeness: Dictionary from check_completeness()
    """
    print("="*80)
    print("BIDS DATA STRUCTURE VALIDATION REPORT")
    print("="*80)
    print()
    
    # Overall statistics
    n_subjects = len(data_summary)
    all_tasks = set()
    for tasks in data_summary.values():
        all_tasks.update(tasks.keys())
    
    print(f"Total subjects: {n_subjects}")
    print(f"Tasks found: {sorted(all_tasks)}")
    print()
    
    # Subject-by-subject summary
    print("="*80)
    print("SUBJECT-BY-SUBJECT SUMMARY")
    print("="*80)
    print()
    
    for subject_id in sorted(data_summary.keys()):
        print(f"\n{subject_id}:")
        print("-" * 40)
        
        tasks = data_summary[subject_id]
        comp = completeness[subject_id]
        
        for task in sorted(tasks.keys()):
            bold_runs = sorted(tasks[task]['bold'])
            event_runs = sorted(tasks[task]['events'])
            
            status = "✓ COMPLETE" if comp[task]['is_complete'] else "✗ INCOMPLETE"
            
            print(f"  {task:20s} {status}")
            print(f"    BOLD runs:  {len(bold_runs):2d}  {bold_runs}")
            print(f"    Event files: {len(event_runs):2d}  {event_runs}")
    
    # EMFL (effloc) specific summary
    print("\n" + "="*80)
    print("EMFL (effloc) TASK SUMMARY")
    print("="*80)
    print()
    
    effloc_subjects = []
    for subject_id in sorted(data_summary.keys()):
        if 'effloc' in data_summary[subject_id]:
            comp = completeness[subject_id]['effloc']
            status = "✓ COMPLETE (5/5 runs)" if comp['is_complete'] else f"✗ INCOMPLETE ({comp['n_bold_runs']}/5 runs)"
            print(f"  {subject_id:20s} {status}")
            
            if comp['is_complete']:
                effloc_subjects.append(subject_id)
    
    print(f"\nSubjects with complete EMFL data: {len(effloc_subjects)}")
    print(f"Subject IDs: {effloc_subjects}")
    
    # Check anatomical data
    print("\n" + "="*80)
    print("ANATOMICAL DATA CHECK")
    print("="*80)
    print()
    
    for subject_id in sorted(data_summary.keys()):
        sub_dir = BIDS_DIR / subject_id
        anat_dir = sub_dir / 'anat'
        
        if anat_dir.exists():
            t1w_files = list(anat_dir.glob('*T1w.nii.gz'))
            if t1w_files:
                print(f"  {subject_id:20s} ✓ T1w found ({len(t1w_files)} file(s))")
            else:
                print(f"  {subject_id:20s} ✗ No T1w files")
        else:
            print(f"  {subject_id:20s} ✗ No anat directory")
    
    return effloc_subjects

def save_summary_to_csv(data_summary, completeness, output_path):
    """
    Save summary to CSV file for further analysis.
    
    Args:
        data_summary: Dictionary from scan_bids_directory()
        completeness: Dictionary from check_completeness()
        output_path: Path to save CSV file
    """
    rows = []
    
    for subject_id in sorted(data_summary.keys()):
        for task, comp in completeness[subject_id].items():
            rows.append({
                'subject_id': subject_id,
                'task': task,
                'n_bold_runs': comp['n_bold_runs'],
                'n_event_files': comp['n_event_files'],
                'has_both': comp['has_both'],
                'is_complete': comp['is_complete']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSummary saved to: {output_path}")
    
    return df

def check_event_file_format(data_summary):
    """
    Check the format of event files to ensure they are properly formatted.
    
    Args:
        data_summary: Dictionary from scan_bids_directory()
    """
    print("\n" + "="*80)
    print("EVENT FILE FORMAT CHECK")
    print("="*80)
    print()
    
    # Check one example event file per task
    checked_tasks = set()
    
    for subject_id in sorted(data_summary.keys()):
        for task in sorted(data_summary[subject_id].keys()):
            if task in checked_tasks:
                continue
                
            # Find an event file for this task
            func_dir = BIDS_DIR / subject_id / 'func'
            event_files = list(func_dir.glob(f'*task-{task}*events.tsv'))
            
            if event_files:
                print(f"Checking {task} event file format...")
                print(f"  File: {event_files[0].name}")
                
                try:
                    events_df = pd.read_csv(event_files[0], sep='\t')
                    print(f"  Columns: {list(events_df.columns)}")
                    print(f"  Number of events: {len(events_df)}")
                    print(f"  ✓ Event file is readable\n")
                    
                    checked_tasks.add(task)
                except Exception as e:
                    print(f"  ✗ ERROR reading event file: {e}\n")

def main():
    """Main function to run BIDS validation."""
    global BIDS_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bids-dir", required=True, type=Path,
                        help="Raw BIDS root of the Marvi 2025 EMFL dataset (OpenNeuro ds006179).")
    parser.add_argument("--output-csv", type=Path, default=Path("bids_validation_summary.csv"),
                        help="Where to write the per-subject/task summary CSV (default: ./bids_validation_summary.csv).")
    args = parser.parse_args()
    BIDS_DIR = args.bids_dir

    print("Starting BIDS validation for Marvi et al. 2025 dataset...\n")

    # Check if BIDS directory exists
    if not BIDS_DIR.exists():
        print(f"ERROR: BIDS directory not found at {BIDS_DIR}")
        return
    
    # Scan directory structure
    print("Scanning BIDS directory structure...")
    data_summary = scan_bids_directory()
    
    # Check completeness
    print("\nChecking data completeness...")
    completeness = check_completeness(data_summary)
    
    # Generate report
    effloc_subjects = generate_summary_report(data_summary, completeness)
    
    # Check event file formats
    check_event_file_format(data_summary)
    
    # Save summary to CSV
    output_path = args.output_csv
    if output_path.parent != Path(""):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    df = save_summary_to_csv(data_summary, completeness, output_path)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nSummary statistics:")
    print(f"  - Total subjects: {len(data_summary)}")
    print(f"  - Subjects with complete EMFL data: {len(effloc_subjects)}")
    print(f"  - Tasks identified: {len(df['task'].unique())}")
    print(f"\nNext steps:")
    print(f"  1. Review the validation report above")
    print(f"  2. Check the CSV summary at: {output_path}")
    print(f"  3. Proceed with fMRIprep preprocessing for subjects with complete data")

if __name__ == "__main__":
    main()

