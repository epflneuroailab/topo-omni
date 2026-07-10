"""
Stimulus timing utilities for Pernet 2015 voice localizer dataset.

This module provides functions to extract and parse stimulus timing information
from the TVA_loc.txt file and create event files for GLM analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def load_stimulus_order(tva_loc_path: str) -> List[int]:
    """
    Load the stimulus block order from TVA_loc.txt file.
    
    Parameters
    ----------
    tva_loc_path : str
        Path to the TVA_loc.txt file
        
    Returns
    -------
    List[int]
        List of 61 timepoints representing block order
    """
    with open(tva_loc_path, 'r') as f:
        # Read all lines and filter out empty ones
        lines = [line.strip() for line in f.readlines()]
        # Remove empty lines but keep the structure
        order = []
        for line in lines:
            if line:  # Only add non-empty lines
                order.append(int(line))
            else:
                # If we encounter an empty line, it might be the last line
                # We'll handle this in the validation
                pass
    return order


def parse_stimulus_timing(tva_loc_path: str, tr: float = 2.0, total_duration: float = 620.0, include_final_silence: bool = False) -> pd.DataFrame:
    """
    Parse stimulus timing information and create events DataFrame.
    
    Parameters
    ----------
    tva_loc_path : str
        Path to the TVA_loc.txt file
    tr : float, optional
        Repetition time in seconds, default 2.0
    total_duration : float, optional
        Total experiment duration in seconds, default 620.0 (10min 20sec)
    include_final_silence : bool, optional
        Whether to include the final 140-second silence period, default False
        
    Returns
    -------
    pd.DataFrame
        Events DataFrame with columns: onset, duration, trial_type
    """
    # Load stimulus order
    order = load_stimulus_order(tva_loc_path)
    
    # Block duration in seconds
    block_duration = 8.0
    
    # Create events list
    events = []
    current_time = 0.0
    
    for i, block_num in enumerate(order):
        if block_num == 99:
            # Silence period
            events.append({
                'onset': current_time,
                'duration': block_duration,
                'trial_type': 'silence'
            })
        elif 1 <= block_num <= 20:
            # Vocal block
            events.append({
                'onset': current_time,
                'duration': block_duration,
                'trial_type': 'vocal'
            })
        elif 21 <= block_num <= 40:
            # Non-vocal block
            events.append({
                'onset': current_time,
                'duration': block_duration,
                'trial_type': 'non_vocal'
            })
        else:
            raise ValueError(f"Invalid block number: {block_num}")
        
        current_time += block_duration
    
    # Check if we need to add a final silence period
    if include_final_silence and current_time < total_duration:
        final_silence_duration = total_duration - current_time
        events.append({
            'onset': current_time,
            'duration': final_silence_duration,
            'trial_type': 'silence'
        })
        print(f"Added final silence period: {final_silence_duration:.1f} seconds (from {current_time:.1f}s to {total_duration:.1f}s)")
    elif current_time < total_duration:
        print(f"Note: Final {total_duration - current_time:.1f}s silence period excluded from analysis")
    
    return pd.DataFrame(events)


def create_glm_events(tva_loc_path: str, tr: float = 2.0, include_final_silence: bool = False) -> pd.DataFrame:
    """
    Create events DataFrame optimized for GLM analysis.
    
    Parameters
    ----------
    tva_loc_path : str
        Path to the TVA_loc.txt file
    tr : float, optional
        Repetition time in seconds, default 2.0
    include_final_silence : bool, optional
        Whether to include the final 140-second silence period, default False
        
    Returns
    -------
    pd.DataFrame
        Events DataFrame with columns: onset, duration, trial_type
    """
    events_df = parse_stimulus_timing(tva_loc_path, tr, include_final_silence=include_final_silence)
    
    # Filter out silence periods for GLM analysis
    glm_events = events_df[events_df['trial_type'] != 'silence'].copy()
    
    return glm_events


def get_stimulus_statistics(tva_loc_path: str) -> Dict:
    """
    Get statistics about the stimulus presentation.
    
    Parameters
    ----------
    tva_loc_path : str
        Path to the TVA_loc.txt file
        
    Returns
    -------
    Dict
        Dictionary containing stimulus statistics
    """
    order = load_stimulus_order(tva_loc_path)
    
    # Count different block types
    vocal_blocks = sum(1 for x in order if 1 <= x <= 20)
    non_vocal_blocks = sum(1 for x in order if 21 <= x <= 40)
    silence_blocks = sum(1 for x in order if x == 99)
    
    # Calculate timing
    total_duration = len(order) * 8.0  # 8 seconds per block
    
    stats = {
        'total_blocks': len(order),
        'vocal_blocks': vocal_blocks,
        'non_vocal_blocks': non_vocal_blocks,
        'silence_blocks': silence_blocks,
        'total_duration_seconds': total_duration,
        'total_duration_minutes': total_duration / 60.0,
        'block_duration_seconds': 8.0
    }
    
    return stats


def validate_stimulus_order(tva_loc_path: str) -> bool:
    """
    Validate that the stimulus order file is correct.
    
    Parameters
    ----------
    tva_loc_path : str
        Path to the TVA_loc.txt file
        
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    try:
        order = load_stimulus_order(tva_loc_path)
        
        # Check length - actual data has 60 timepoints (not 61 as originally expected)
        if len(order) != 60:
            print(f"Expected 60 timepoints, got {len(order)}")
            return False
        
        # Check block numbers
        vocal_blocks = sum(1 for x in order if 1 <= x <= 20)
        non_vocal_blocks = sum(1 for x in order if 21 <= x <= 40)
        silence_blocks = sum(1 for x in order if x == 99)
        
        # Verify block counts
        if vocal_blocks + non_vocal_blocks + silence_blocks != len(order):
            print(f"Invalid block numbers detected")
            return False
        
        # Verify expected block counts
        if vocal_blocks != 20 or non_vocal_blocks != 20 or silence_blocks != 20:
            print(f"Incorrect block distribution:")
            print(f"  Vocal blocks: {vocal_blocks} (expected 20)")
            print(f"  Non-vocal blocks: {non_vocal_blocks} (expected 20)")
            print(f"  Silence blocks: {silence_blocks} (expected 20)")
            return False
        
        print(f"Status: Validation passed: {vocal_blocks} vocal + {non_vocal_blocks} non-vocal + {silence_blocks} silence = {len(order)} total blocks")
        return True
        
    except Exception as e:
        print(f"Error validating stimulus order: {str(e)}")
        return False


def test_event_timing_validation(events_df: pd.DataFrame, include_final_silence: bool = False) -> bool:
    """
    Validate event timing DataFrame.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events DataFrame to validate
    include_final_silence : bool, optional
        Whether final silence period is included, default False
        
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    try:
        # Basic DataFrame validation
        required_columns = ['onset', 'duration', 'trial_type']
        if not all(col in events_df.columns for col in required_columns):
            print("Error: Missing required columns")
            return False
        
        # Check for NaN values
        if events_df.isnull().any().any():
            print("Error: NaN values detected in events DataFrame")
            return False
        
        # Validate trial types
        valid_types = {'vocal', 'non_vocal', 'silence'} if include_final_silence else {'vocal', 'non_vocal'}
        if not set(events_df['trial_type'].unique()).issubset(valid_types):
            print("Error: Invalid trial types detected")
            return False
        
        # Validate timing
        if not np.all(events_df['onset'] >= 0):
            print("Error: Negative onset times detected")
            return False
        
        if not np.all(events_df['duration'] > 0):
            print("Error: Non-positive durations detected")
            return False
        
        # Check for temporal overlap
        end_times = events_df['onset'] + events_df['duration']
        next_onsets = events_df['onset'].shift(-1)
        if not np.all((next_onsets - end_times)[:-1] >= 0):
            print("Error: Temporal overlap detected between events")
            return False
        
        print("Status: Event timing validation passed")
        return True
        
    except Exception as e:
        print(f"Error validating event timing: {str(e)}")
        return False


def validate_glm_events(events_df: pd.DataFrame) -> bool:
    """
    Validate GLM events DataFrame.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        GLM events DataFrame to validate
        
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    try:
        # Basic validation
        if not test_event_timing_validation(events_df, include_final_silence=False):
            return False
        
        # GLM-specific validation
        if 'silence' in events_df['trial_type'].unique():
            print("Error: Silence periods should be excluded from GLM events")
            return False
        
        # Verify vocal vs non-vocal balance
        n_vocal = sum(events_df['trial_type'] == 'vocal')
        n_non_vocal = sum(events_df['trial_type'] == 'non_vocal')
        
        if n_vocal != n_non_vocal:
            print(f"Error: Unbalanced design - {n_vocal} vocal vs {n_non_vocal} non-vocal blocks")
            return False
        
        if n_vocal != 20 or n_non_vocal != 20:
            print(f"Error: Incorrect number of blocks - expected 20 each, got {n_vocal} vocal and {n_non_vocal} non-vocal")
            return False
        
        print("Status: Event timing validation passed")
        return True
        
    except Exception as e:
        print(f"Error validating GLM events: {str(e)}")
        return False


def validate_tva_loc_file(tva_loc_path: str) -> bool:
    """
    Validate TVA_loc.txt file.
    
    Parameters
    ----------
    tva_loc_path : str
        Path to the TVA_loc.txt file
        
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    try:
        # File existence check
        if not Path(tva_loc_path).exists():
            print(f"Error: File not found - {tva_loc_path}")
            return False
        
        # Content validation
        if not validate_stimulus_order(tva_loc_path):
            print("Error: Invalid stimulus order")
            return False
        
        # Event timing validation
        events_df = parse_stimulus_timing(tva_loc_path)
        if not test_event_timing_validation(events_df):
            print("Error: Invalid event timing")
            return False
        
        # GLM events validation
        glm_events = create_glm_events(tva_loc_path)
        if not validate_glm_events(glm_events):
            print("Error: Invalid GLM events")
            return False
        
        print("Status: Stimulus order file validated")
        return True
        
    except Exception as e:
        print(f"Error validating TVA_loc file: {str(e)}")
        return False


if __name__ == '__main__':
    # Example usage
    tva_loc_path = "data/TVA_loc.txt"
    
    # Validate the file
    if validate_tva_loc_file(tva_loc_path):
        print("Status: Stimulus order file validated")
    else:
        print("Error: Invalid stimulus order file") 