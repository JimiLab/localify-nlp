from typing import List
import yaml
import json
import bidict
from random import shuffle
from typing import Tuple
from math import ceil
from random import sample
import numpy as np
import torch
import gc
from transformers import AutoTokenizer, Gemma3ForCausalLM


torch.cuda.empty_cache()
gc.collect()


class GemmaWrapper:

    def __init__(self, model, tokenizer, system_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def get_messages(self, system_message, user_message):
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message},]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_message},]
                },
            ],
        ]
        return messages

    def get_response(self, prompt):
        messages = self.get_messages(self.system_prompt, prompt)

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        generation = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)
        generation = generation[0][input_len:]

        decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
        #print(decoded)
        return decoded


class Recommender:

    def __init__(self, artists, prompt):
        # essentially json of {artist_id: artist_data (genres tags, wiki, etc)}
        self.artists = artists
        self.prompt = prompt

    def recommend(self, seed_ids: List[str], candidate_ids: List[str], use_genres=False) -> List[str]:
        raise NotImplementedError(f'{self.__class__} class is an interface and not intended for instantiation.')

    def parse_output(self, output, candidate_dict):
        start_index = output.rfind('<') + 1
        end_index = output.rfind('>')
        if start_index == -1 or end_index == -1:
            return None
        cleaned = output[start_index:end_index]
        try:
            iids = map(int, cleaned.split(','))
            return [candidate_dict.inverse[i] for i in iids if i in candidate_dict.inverse]
        except Exception as e:
            return None


class InternalRecommender(Recommender):

    def __init__(self, artists, prompt, gemma_wrapper):
        super().__init__(artists, prompt)
        self.gw = gemma_wrapper

    def recommend(self, seed_ids: List[str], candidate_ids: List[str], use_genres=False) -> List[str]:
        #print('Seeds: ')
        #print(seed_ids)
        #print('Candidates: ')
        #print(candidate_ids)
        iids = list(range(len(candidate_ids)))
        candidate_dict = bidict.bidict({candidate_ids[i]: iids[i] for i in range(len(candidate_ids))})
        seed_names = '\n'.join([self.artists[_id]['name'] + ('' if not use_genres else f" ({', '.join(self.artists[_id]['genres'])})") for _id in seed_ids])
        candidate_names = '\n'.join([self.artists[_id]['name'] + ('' if not use_genres else f" ({', '.join(self.artists[_id]['genres'])})") for _id in candidate_ids])
        candidate_dict_text = '\n'.join([f"{candidate_dict[_id]}: {self.artists[_id]['name']}" for _id in candidate_ids])
        _prompt = self.prompt.format(seeds=seed_names, candidates=candidate_names, candidate_key=candidate_dict_text)
        text = self.gw.get_response(_prompt)
        #print("Response Text: ")
        #print(text, end="\n\n")
        return self.parse_output(text, candidate_dict)


def calc_auc_score(rank_relevance):
    """
    usage : result = model.calc_auc_score([1,0,1,0,0,0,1,0,0,0,0,0])
    :param rank_relevance: list of 1s (relevant) and 0s (not relevant)
    :return: AUC score between 0 and 1. 0.5 is random. 1.0 is perfect (all relevant items at the top.)
    """
    num_true = sum(rank_relevance)
    num_false = len(rank_relevance) - num_true

    if num_true == 0 or num_false == 0:
        return -1

    tpr = 0
    total = 0
    for val in rank_relevance:
        if val:
            tpr += 1
        else:
            total += tpr

    auc = total / (num_true * num_false)
    return auc


class Evaluator:

    def __init__(self, recommender, artist_ids, seed_lists):
        self.recommender = recommender
        self.artist_ids = artist_ids
        self.seed_lists = seed_lists

    def do_trial(self, seed_ids, use_genres=False):
        split = ceil(len(seed_ids) * 0.5)
        shuffle(seed_ids)
        masked = seed_ids[split:]
        masked = masked[:10] if len(masked) >= 10 else masked
        unmasked = seed_ids[:split]
        unmasked = unmasked[:10] if len(unmasked) >= 10 else unmasked

        potential_distractors = [_id for _id in self.artist_ids if _id not in seed_ids]
        distractors = sample(potential_distractors, len(masked))

        candidates = masked + distractors
        shuffle(candidates)
        results = self.recommender.recommend(unmasked, candidates, use_genres)
        if results is None:
            return -1

        relevances = [1 if res in masked else 0 for res in results]

        # ChatGPT doesn't always return the full list, so pad with a sampled list of 1s and 0s according to the appropriate proportions
        size_difference = len(candidates) - len(relevances)
        #print('Difference', size_difference)
        if size_difference > 0:
            num_pos = len([x for x in relevances if x == 1])
            prop_pos = num_pos / len(relevances)
            pad_neg = ceil(size_difference * prop_pos)
            pad_pos = size_difference - pad_neg
            padding = [0 for i in range(pad_neg)] + [1 for i in range(pad_pos)]
            shuffle(padding)
            #print('Padding:', padding)
            relevances += padding

        #print(relevances)
        return calc_auc_score(relevances)

    def eval_model(self, use_genres=False):
        scores = []
        for seed_ids in self.seed_lists:
            score = self.do_trial(seed_ids, use_genres)
            print('Score:', score)
            if score != -1:
                scores.append(score)
        print(scores)
        return np.mean(scores)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

artists = load_json("nlp_artists_filtered.json")
seeds = load_json("nlp_seeds_anonymized.json")
artist_ids = [k for k in artists.keys()]

with open('../../conf.yaml') as file:
    conf = yaml.safe_load(file)

ckpt = "google/gemma-3-4b-it"
gemma_model = Gemma3ForCausalLM.from_pretrained(
    ckpt,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_auth_token=conf['gemma-key'],
    low_cpu_mem_usage=True,
)

gemma_tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-4b-it",
    use_auth_token=conf['gemma-key'],
)

system_prompt_3 = "You are an expert in music recommendation. Your specialty is in ranking a list of artists by how similar each one is to a different set of artists that someone already knows."
prompt_3 = """
You are presented with a client who frequently listens to the following artists:
{seeds}

You are asked to use your expert knowledge of these artists to rank the following artists (on which you are also an expert) in order from most recommended to least recommended:
{candidates}

You must present these recommendations in a very specific way. Each candidate artist that you are recommending has an integer ID associated with them. The key is as follows:
{candidate_key}

You must finish your response by listing your recommendations only using these ids, separated by commas, and surrounded by <>.
An example recommendation would be as follows:
<#,#,#,#,#,#,#,#,#,#,...>
With each hashtag replaced by the artist you recommend in that position.
Provide me with only the ranked list of integer IDs associated with the ranking, do not enumerate your thought process.
"""
gw_3 = GemmaWrapper(gemma_model, gemma_tokenizer, system_prompt_3)
gemma_3 = InternalRecommender(artists, prompt_3, gw_3)
evaluator_gemma_3 = Evaluator(gemma_3, artist_ids, seeds)
gemma_3_score = evaluator_gemma_3.eval_model(use_genres=True)
print("Experiment 2 Gemma Score:", gemma_3_score)
