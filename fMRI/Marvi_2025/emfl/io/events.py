#!/usr/bin/env python3
"""
Event File Parser for Marvi et al. 2025 fMRI Analysis

This module provides functions to parse BIDS-formatted event files and create
design matrices suitable for nilearn GLM analysis.

Key Features:
- Reads TSV event files with onset, duration, trial_type columns
- Handles special case of effloc task with separate visual/auditory event files
- Filters out fixation trials (baseline)
- Creates design matrices compatible with nilearn FirstLevelModel
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_event_file(event_file_path: str) -> pd.DataFrame:
    """
    Load a BIDS event file from TSV format.
    
    Parameters
    ----------
    event_file_path : str
        Path to the event TSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: onset, duration, trial_type, response_time
        
    Notes
    -----
    The function automatically handles 'NA' values in response_time column.
    """
    events = pd.read_csv(event_file_path, sep='\t')
    
    # Replace 'NA' with NaN for response_time
    if 'response_time' in events.columns:
        events['response_time'] = pd.to_numeric(events['response_time'], errors='coerce')
    
    return events


def filter_fixation(events: pd.DataFrame) -> pd.DataFrame:
    """
    Remove fixation trials from events DataFrame.
    
    Fixation periods serve as implicit baseline in GLM and should not be
    explicitly modeled as a condition.
    
    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame with trial_type column
        
    Returns
    -------
    pd.DataFrame
        Filtered events without fixation trials
    """
    return events[events['trial_type'] != 'fixation'].copy()


def get_effloc_events(subject_dir: Path, run_num: str, 
                      modality: str = 'visual') -> pd.DataFrame:
    """
    Load effloc task events for either visual or auditory modality.
    
    The effloc task presents audio-visual stimuli simultaneously, with separate
    event files for visual and auditory conditions.
    
    Parameters
    ----------
    subject_dir : Path
        Path to subject's BIDS directory (e.g., /path/to/sub-kaneff01)
    run_num : str
        Run number (e.g., '001', '002', etc.)
    modality : str, optional
        Either 'visual' or 'auditory', default 'visual'
        
    Returns
    -------
    pd.DataFrame
        Events DataFrame for the specified modality
        
    Raises
    ------
    ValueError
        If modality is not 'visual' or 'auditory'
    FileNotFoundError
        If event file does not exist
    """
    if modality not in ['visual', 'auditory']:
        raise ValueError(f"modality must be 'visual' or 'auditory', got '{modality}'")
    
    subject_id = subject_dir.name
    
    if modality == 'visual':
        task_name = 'efflocVisualConditions'
    else:
        task_name = 'efflocAuditoryConditions'
    
    event_file = subject_dir / 'func' / f'{subject_id}_task-{task_name}_run-{run_num}_events.tsv'
    
    if not event_file.exists():
        raise FileNotFoundError(f"Event file not found: {event_file}")
    
    events = load_event_file(str(event_file))
    return filter_fixation(events)


def get_standard_task_events(subject_dir: Path, task_name: str, 
                             run_num: str) -> pd.DataFrame:
    """
    Load events for standard tasks (non-effloc).
    
    Parameters
    ----------
    subject_dir : Path
        Path to subject's BIDS directory
    task_name : str
        Task name (e.g., 'lang', 'speech', 'eploc', 'foss', 'spwm')
    run_num : str
        Run number (e.g., '001', '002', etc.)
        
    Returns
    -------
    pd.DataFrame
        Events DataFrame with fixation trials removed
        
    Raises
    ------
    FileNotFoundError
        If event file does not exist
    """
    subject_id = subject_dir.name
    event_file = subject_dir / 'func' / f'{subject_id}_task-{task_name}_run-{run_num}_events.tsv'
    
    if not event_file.exists():
        raise FileNotFoundError(f"Event file not found: {event_file}")
    
    events = load_event_file(str(event_file))
    return filter_fixation(events)


def filter_runs_by_split(run_numbers: List[str], run_split: str = 'all') -> List[str]:
    """
    Filter run numbers based on even/odd split for cross-validation.
    
    This function supports the run-splitting strategy used in the original
    Marvi et al. (2025) analysis for independent validation of fROIs.
    
    Parameters
    ----------
    run_numbers : List[str]
        List of run numbers (e.g., ['001', '002', '003', '004', '005'])
    run_split : str
        Type of run split to apply:
        - 'all': Use all runs (no filtering)
        - 'even': Use only even-numbered runs (002, 004, etc.)
        - 'odd': Use only odd-numbered runs (001, 003, 005, etc.)
        
    Returns
    -------
    List[str]
        Filtered list of run numbers
        
    Examples
    --------
    >>> runs = ['001', '002', '003', '004', '005']
    >>> filter_runs_by_split(runs, 'odd')
    ['001', '003', '005']
    >>> filter_runs_by_split(runs, 'even')
    ['002', '004']
    >>> filter_runs_by_split(runs, 'all')
    ['001', '002', '003', '004', '005']
    
    Notes
    -----
    Run splitting enables cross-validation:
    - Define fROIs on odd runs, test on even runs
    - Define fROIs on even runs, test on odd runs
    - Average results for robust estimates
    """
    if run_split not in ['all', 'even', 'odd']:
        raise ValueError(f"run_split must be 'all', 'even', or 'odd', got '{run_split}'")
    
    if run_split == 'all':
        return run_numbers
    
    # Convert run numbers to integers for even/odd check
    filtered_runs = []
    for run in run_numbers:
        run_int = int(run)
        if run_split == 'even' and run_int % 2 == 0:
            filtered_runs.append(run)
        elif run_split == 'odd' and run_int % 2 == 1:
            filtered_runs.append(run)
    
    return filtered_runs


def get_all_runs_for_task(subject_dir: Path, task_name: str, 
                          modality: Optional[str] = None,
                          run_split: str = 'all') -> List[pd.DataFrame]:
    """
    Load run events for a given task, with optional even/odd filtering.
    
    Parameters
    ----------
    subject_dir : Path
        Path to subject's BIDS directory
    task_name : str
        Task name (e.g., 'effloc', 'lang', 'speech', etc.)
    modality : str, optional
        For effloc task only: 'visual' or 'auditory'
    run_split : str, optional
        Type of run split: 'all', 'even', or 'odd'. Default: 'all'
        
    Returns
    -------
    List[pd.DataFrame]
        List of events DataFrames, one per run (filtered by run_split)
        
    Examples
    --------
    >>> # Get all visual effloc runs
    >>> events = get_all_runs_for_task(subject_dir, 'effloc', modality='visual')
    >>> 
    >>> # Get only odd runs for cross-validation
    >>> events_odd = get_all_runs_for_task(subject_dir, 'effloc', 
    ...                                     modality='visual', run_split='odd')
    """
    # Determine expected number of runs per task
    expected_runs = {
        'effloc': 5,
        'eploc': 2,
        'foss': 4,
        'lang': 2,
        'speech': 2,
        'spwm': 2
    }
    
    n_runs = expected_runs.get(task_name, 2)  # Default to 2 if unknown
    
    # Generate all run numbers
    all_run_numbers = [f"{run_idx:03d}" for run_idx in range(1, n_runs + 1)]
    
    # Filter runs based on split type
    run_numbers = filter_runs_by_split(all_run_numbers, run_split)
    
    events_list = []
    
    for run_num in run_numbers:
        try:
            if task_name == 'effloc':
                if modality is None:
                    raise ValueError("modality must be specified for effloc task")
                events = get_effloc_events(subject_dir, run_num, modality)
            else:
                events = get_standard_task_events(subject_dir, task_name, run_num)
            
            events_list.append(events)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
    
    return events_list


def create_design_matrix_inputs(events: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Convert events DataFrame to format suitable for nilearn make_first_level_design_matrix.
    
    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame with onset, duration, trial_type columns
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with 'onset', 'duration', 'trial_type' arrays suitable for nilearn
        
    Notes
    -----
    This format is directly compatible with nilearn.glm.first_level.make_first_level_design_matrix
    """
    return {
        'onset': events['onset'].values,
        'duration': events['duration'].values,
        'trial_type': events['trial_type'].values
    }


def get_unique_conditions(events: pd.DataFrame) -> List[str]:
    """
    Get sorted list of unique conditions in events DataFrame.
    
    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame with trial_type column
        
    Returns
    -------
    List[str]
        Sorted list of unique trial types (excluding fixation)
    """
    conditions = events['trial_type'].unique()
    conditions = [c for c in conditions if c != 'fixation']
    return sorted(conditions)


def summarize_events(events: pd.DataFrame, task_name: str, run_num: str) -> None:
    """
    Print a summary of event timing and conditions.
    
    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame
    task_name : str
        Name of the task
    run_num : str
        Run number
    """
    print(f"\n{'='*60}")
    print(f"Task: {task_name} | Run: {run_num}")
    print(f"{'='*60}")
    print(f"Total events: {len(events)}")
    print(f"Duration: {events['onset'].max() + events['duration'].max():.1f}s")
    print(f"\nConditions:")
    
    for condition in get_unique_conditions(events):
        condition_events = events[events['trial_type'] == condition]
        n_trials = len(condition_events)
        total_duration = condition_events['duration'].sum()
        print(f"  {condition:20s}: {n_trials:2d} trials, {total_duration:6.1f}s total")


if __name__ == "__main__":
    """
    Example usage and testing of the event file parser.
    """
    import sys
    
    # Test with sub-kaneff01
    bids_root = Path("/work/upschrimpf1/mehrer/datasets/Marvi_2025_efficient_fMRI_localizer/orig_data")
    subject_dir = bids_root / "sub-kaneff01"
    
    if not subject_dir.exists():
        print(f"Error: Subject directory not found: {subject_dir}")
        sys.exit(1)
    
    print("="*70)
    print("Event File Parser - Testing")
    print("="*70)
    
    # Test 1: Load effloc visual conditions
    print("\nTest 1: Loading effloc visual conditions (run 1)")
    events = get_effloc_events(subject_dir, '001', modality='visual')
    summarize_events(events, 'effloc_visual', '001')
    
    # Test 2: Load effloc auditory conditions
    print("\nTest 2: Loading effloc auditory conditions (run 1)")
    events = get_effloc_events(subject_dir, '001', modality='auditory')
    summarize_events(events, 'effloc_auditory', '001')
    
    # Test 3: Load lang task
    print("\nTest 3: Loading lang task (run 1)")
    events = get_standard_task_events(subject_dir, 'lang', '001')
    summarize_events(events, 'lang', '001')
    
    # Test 4: Load all runs for lang task
    print("\nTest 4: Loading all lang runs")
    all_lang_events = get_all_runs_for_task(subject_dir, 'lang')
    print(f"Loaded {len(all_lang_events)} runs for lang task")
    
    # Test 5: Create design matrix inputs
    print("\nTest 5: Creating design matrix inputs")
    dm_inputs = create_design_matrix_inputs(events)
    print(f"Design matrix format:")
    print(f"  - onset: {dm_inputs['onset'].shape}")
    print(f"  - duration: {dm_inputs['duration'].shape}")
    print(f"  - trial_type: {dm_inputs['trial_type'].shape}")
    print(f"  - Unique conditions: {np.unique(dm_inputs['trial_type'])}")
    
    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)



