import os
import json 
from tqdm import tqdm
from glob import glob

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# read video and return duration
def get_video_duration_ffprobe(video_path):
    import shlex
    import subprocess

    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(video_path)}"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.decode()}")
    return float(result.stdout.decode().strip())

def get_num_chunks(video_lists):
    video_to_num_chunks = {}
    for video_path in video_lists:
        video_id = os.path.basename(video_path)
        video_name = video_id.split("-")[-1].split("_")[0]
        chunk_num = int(video_id.split("-")[-1].split("_")[2].replace(".mp4", ""))
        if video_name not in video_to_num_chunks:
            video_to_num_chunks[video_name] = chunk_num

        if chunk_num > video_to_num_chunks[video_name]:
            video_to_num_chunks[video_name] = chunk_num
    return video_to_num_chunks
        

if __name__ == "__main__":

    video_lists = read_json("task-alignvideo/clustering_v2/clips_manifest.json")
    video_to_num_chunks = get_num_chunks(video_lists)

    merged_clusters = read_json("task-alignvideo/clustering_v2/merged_clusters_tvals_v2.json")

    videos = glob("task-alignvideo/*.mp4")
    video_durations = {}
    for video in tqdm(videos):
        video_id = video.split("/")[-1].split("-")[-1].replace(".mp4", "")
        duration = get_video_duration_ffprobe(video)
        num_chunks = int(duration // 2)
        video_durations[video_id] = {
            "duration": duration,
            "num_chunks": num_chunks,
        }

        assert video_to_num_chunks[video_id] <= video_durations[video_id]["num_chunks"], \
            f"Video {video_id} has {video_to_num_chunks[video_id]} chunks in manifest but only {video_durations[video_id]['num_chunks']} possible chunks based on duration {duration:.2f}s"

    for cluster in merged_clusters:
        cluster["video_info"] = {}
        for video_id in cluster["video_ids"]:
            video_name = video_id.split("-")[-1].split("_")[0]
            chunk_num = int(video_id.split("-")[-1].split("_")[2].replace(".mp4", ""))
            cluster["video_info"][video_id] = {
                "video_name": video_name,
                "start_time": chunk_num * 2,
                "end_time": (chunk_num + 1) * 2,
            }

    with open("task-alignvideo/clustering_v2/merged_clusters_tvals_with_video_info_v2.json", "w") as f:
        json.dump(merged_clusters, f, indent=4)


