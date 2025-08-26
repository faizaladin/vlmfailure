import json


with open('llava_finetune.json', 'r') as f:
    data = json.load(f)
    print(f"Total number of entries: {len(data)}")

    # Extract trajectory (folder) names from image paths
    def get_traj(entry):
        # expects path like paired_frames/<traj>/<frame>.png
        return entry['image'].split('/')[1]

    # Get all unique trajectories, sorted
    trajs = sorted(set(get_traj(entry) for entry in data))
    last_10_trajs = trajs[-10:]

    # Count entries belonging to last 10 trajectories
    last_10_count = sum(1 for entry in data if get_traj(entry) in last_10_trajs)
    print(f"Entries in the last 10 trajectories: {last_10_count}")
