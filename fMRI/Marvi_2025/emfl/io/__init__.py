"""
Input/Output Utilities
======================

Utilities for handling BIDS-formatted data and event files.
"""

from emfl.io.events import (
    load_event_file,
    filter_fixation,
    get_effloc_events,
    get_standard_task_events,
    get_all_runs_for_task,
    create_design_matrix_inputs,
    filter_runs_by_split,
)

__all__ = [
    'load_event_file',
    'filter_fixation',
    'get_effloc_events',
    'get_standard_task_events',
    'get_all_runs_for_task',
    'create_design_matrix_inputs',
    'filter_runs_by_split',
]

