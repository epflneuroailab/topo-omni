import json 

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

clusters_to_merge = {
    "space": {0,1,2,3,4},
    "animals_1": {5,6,7},
    "cartoon": {8,9,10,11,12},
    "nature_1": {13},
    "war": {14,15,16},
    "natural_disasters": {17,18,19},
    "animals_2": {20},
    "social": {21,22,23,24},
    "nature_2": {25},
    "crowds": {26,27,28},
    "colors": {29},
    "nature_3": {30,31,32},
    "faces_1": {33,34,35,36,37,38},
    "faces_2": {39, 40},
    "faces_3": {41},
    "faces_4": {42},
    "faces_5": {43,44,45,46},
    "faces_6": {47},
    "faces_7": {49},
    "faces_8": {50},
    "faces_9": {48},
    "paintings": {51,52,53},
}


if __name__ == "__main__":

    # print(list(clusters_to_merge.keys()))
    # exit()

    # Read the original cluster assignments
    original_clusters = read_json("task-alignvideo/clustering_v2/clusters_tvals.json")

    # Create a new dictionary to hold the merged cluster assignments
    merged_clusters = []

    # Iterate through the clusters to merge and assign new cluster IDs
    for idx, (cluster_label, cluster_ids) in enumerate(clusters_to_merge.items()):
        video_ids = []
        for row in original_clusters:
            cluster_id = row["cluster_id"]
            if cluster_id in cluster_ids:
                video_ids.extend(row["video_ids"])
            
        merged_clusters.append({
            "cluster_id": idx+1,
            "cluster_label": cluster_label,
            "size": len(video_ids),
            "video_ids": video_ids
        })


    # Save the merged cluster assignments to a new JSON file
    save_path = "task-alignvideo/clustering_v2/merged_clusters_tvals_v3.json"
    with open(save_path, "w") as f:
        json.dump(merged_clusters, f, indent=4)

    print(f"Merged cluster assignments saved to '{save_path}'")