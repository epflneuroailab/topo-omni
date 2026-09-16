top_k=1
cluster_id=32

python -m src.experiments.spacetop_discovery.contrast --cluster_id $cluster_id --topk $top_k
python -m src.experiments.spacetop_discovery.plot_selectivity --cluster_id $cluster_id --topk $top_k