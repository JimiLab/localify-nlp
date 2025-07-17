import json
import numpy as np


tracker = json.load(open("experiments/Score Tracker.json"))
scores_list = []
for value in tracker.values():
    for scores in value:
        scores_list.append(scores)

tracker_flat = {}
for i in range(len(scores_list[0])):
    collected_scores = []
    for scores in scores_list:
        collected_scores.append(scores[i])
    filtered_scores = list(filter(lambda score: score != -1, collected_scores))
    info = {
        "mean_success": np.mean(filtered_scores),
        "sd_success": np.std(filtered_scores),
        "min_success": min(filtered_scores),
        "max_success": max(filtered_scores),
        "failed": len(collected_scores) - len(filtered_scores),
        "scores": collected_scores
    }
    tracker_flat[i] = info

artists = json.load(open("nlp_artists_filtered.json"))
seeds = json.load(open("nlp_seeds_anonymized.json"))
for key, value in tracker_flat.items():
    seed_set = seeds[key]
    pops = [int(artists[seed]["spotify_popularity"]) for seed in seed_set]
    tracker_flat[key]["popularity"] = {
        "mean": np.mean(pops),
        "sd": np.std(pops),
        "max": max(pops),
        "min": min(pops)
    }

csv = ""
for value in tracker_flat.values():
    csv += f"{value['mean_success']}, {value['popularity']['mean']}\n"
open("out.csv", 'w').write(csv.strip())

open("experiments/Score Tracker Flat.json", 'w').write(json.dumps(tracker_flat, indent=4))
