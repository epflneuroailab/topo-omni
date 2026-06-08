import os
import numpy as np

if __name__ == "__main__":

    n_span = 3
    for span_idx in range(50-n_span+1):
        groups = np.arange(span_idx, min(span_idx+n_span, 50))
        groups_str = ','.join(list(map(str, groups)))
        command = f"python -m eval.run.run_clusters_iou --groups \"{groups_str}\" --top-percentile 5.0 --group-name G3"
        # command = f"python -m eval.run.run_clusters_iou --groups \"{groups_str}\" --threshold-mode fdr"
        os.system(command)

