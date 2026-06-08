top_k=1
cluster_id=32

python -m src.eval.analysis.contrast_spacetop --cluster_id $cluster_id --topk $top_k
python -m src.visualize.spacetop_selectivity --cluster_id $cluster_id --topk $top_k