import json
import numpy as np


seeds = json.load(open("nlp_seeds_anonymized.json"))
artists = json.load(open("nlp_artists_filtered.json"))

first_thresholds = []
second_thresholds = []

for user_seeds in seeds:
    user_seeds_sorted = sorted(user_seeds, key=lambda artist_id: int(artists[artist_id]["spotify_popularity"]))
    first_index = round(0.333 * len(user_seeds_sorted))
    second_index = round(0.666 * len(user_seeds_sorted))
    first_thresholds.append(int(artists[user_seeds_sorted[first_index]]["spotify_popularity"]))
    second_thresholds.append(int(artists[user_seeds_sorted[second_index]]["spotify_popularity"]))

first_threshold_final = np.mean(first_thresholds)
second_threshold_final = np.mean(second_thresholds)

print(first_thresholds)
print(first_threshold_final, np.std(first_thresholds))

print(second_thresholds)
print(second_threshold_final, np.std(second_thresholds))

seeds_split = []
for user_seeds in seeds:
    user_seeds_split = {"low": [], "med": [], "high": []}
    for user_seed in user_seeds:
        pop = int(artists[user_seed]["spotify_popularity"])
        if pop < first_threshold_final:
            user_seeds_split["low"].append(user_seed)
        elif pop < second_threshold_final:
            user_seeds_split["med"].append(user_seed)
        else:
            user_seeds_split["high"].append(user_seed)
    print(len(user_seeds_split["low"]), len(user_seeds_split["med"]), len(user_seeds_split["high"]))
    seeds_split.append(user_seeds_split)

open("nlp_seeds_split.json", 'w').write(json.dumps(seeds_split, indent=4))

seeds_categorized = {"low": [], "med": [], "high": []}
for user_seeds_split in seeds_split:
    for key in seeds_categorized.keys():
        if len(user_seeds_split[key]) >= 10:
            seeds_categorized[key].append(user_seeds_split[key])

print("Users Available:")
for key in seeds_categorized.keys():
    print(f"{len(seeds_categorized[key])} in {key.capitalize()}")

open("nlp_seeds_categorized.json", 'w').write(json.dumps(seeds_categorized, indent=4))
