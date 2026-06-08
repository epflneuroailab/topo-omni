import os
import json
import numpy as np
from glob import glob

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":

    dirpath = "spacetop_embeddings_v2"

    path_embeddings = glob(os.path.join(dirpath, "*nemotron_embeddings.npz"))
    path_manifests = glob(os.path.join(dirpath, "*.json"))

    print(f"Found {len(path_embeddings)} embedding files and {len(path_manifests)} manifest files.")

    out_manifest = "task-alignvideo/clustering_v2/clips_manifest.json"
    out_embeddings = "task-alignvideo/clustering_v2/video_clip_vectors.npy"

    os.makedirs(os.path.dirname(out_embeddings), exist_ok=True)

    manifest = []
    all_embeddings = []
    for emb_path in path_embeddings:
        data = np.load(emb_path)
        embeddings = data["embeddings"] # shape (num_clips, embedding_dim)
        chunk_paths = data["chunk_paths"]

        all_embeddings.extend(embeddings)
        manifest.extend(chunk_paths)

    all_embeddings = np.stack(all_embeddings, axis=0)
    print(f"Compiled embeddings shape: {all_embeddings.shape}")

    np.save(out_embeddings, all_embeddings)
    write_json(manifest, out_manifest)



        
        