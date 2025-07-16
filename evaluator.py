import json

import numpy as np
from math import ceil
from random import shuffle, sample

from experiments.results_container import ResultsContainer
from utils import calc_auc_score


class Evaluator:

    def __init__(self, recommender, artist_ids, seed_lists):
        self.recommender = recommender
        self.artist_ids = artist_ids
        self.seed_lists = seed_lists

    def do_trial(self, seed_ids, use_genres, use_wiki, embed_seeds, contrast_num):
        split = ceil(len(seed_ids) * 0.5)
        shuffle(seed_ids)
        masked = seed_ids[split:]
        shuffle(masked)
        masked = masked[:30] if len(masked) >= 30 else masked
        unmasked = seed_ids[:split]

        potential_distractors = [_id for _id in self.artist_ids if _id not in seed_ids]
        distractors = sample(potential_distractors, len(masked))

        candidates = masked + distractors
        shuffle(candidates)
        results = self.recommender.recommend(unmasked, candidates, use_genres, use_wiki, embed_seeds, contrast_num)
        if results is None:
            return -1

        relevances = [1 if res in masked else 0 for res in results]

        # ChatGPT doesn't always return the full list, so pad with a sampled list of 1s and 0s according to the appropriate proportions
        size_difference = len(candidates) - len(relevances)
        print('Difference', size_difference)
        if size_difference > 0:
            num_pos = len([x for x in relevances if x == 1])
            prop_pos = num_pos / len(relevances)
            pad_neg = ceil(size_difference * prop_pos)
            pad_pos = size_difference - pad_neg
            padding = [0 for i in range(pad_neg)] + [1 for i in range(pad_pos)]
            shuffle(padding)
            print('Padding:', padding)
            relevances += padding

        print(relevances)
        return calc_auc_score(relevances)

    def eval_model(self, result_path, experiment_name, use_genres=False, use_wiki=False, embed_seeds=False, contrast_num=0):
        if contrast_num and not embed_seeds:
            raise ValueError("Cannot use contrasting for the embedded descriptions of embedding is disabled.")

        scores = []
        for seed_ids in self.seed_lists:
            score = self.do_trial(seed_ids, use_genres, use_wiki, embed_seeds, contrast_num)
            print('Score:', score)
            scores.append(score)
        print(scores)
        scores_successful = [score for score in scores if score != -1]

        score_tracker = {}
        try:
            score_tracker = json.load(open("Score Tracker.json"))
        except (FileNotFoundError, json.JSONDecodeError) as ignored:
            print("Failed to read score tracker file. Writing a new one.")

        score_key = experiment_name.lower().replace(' ', '_')
        if score_key in score_tracker:
            score_tracker[score_key].append(scores)
        else:
            score_tracker[score_key] = []
            score_tracker[score_key].append(scores)

        open("Score Tracker.json", 'w').write(json.dumps(score_tracker, indent=4))

        results = ResultsContainer(
            mean=np.mean(scores_successful),
            sd=np.std(scores_successful),
            min=np.min(scores_successful),
            max=np.max(scores_successful)
        )
        results.to_file(result_path, experiment_name)
        return results
