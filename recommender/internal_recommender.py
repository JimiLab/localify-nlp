from random import sample
from typing import List

import bidict

from recommender.base_recommender import Recommender


class InternalRecommender(Recommender):

    def __init__(self, artists, prompt, gemma_wrapper, seeds=None, embed_prompt=None, system_prompt=None, system_embed_prompt=None):
        super().__init__(artists, prompt, seeds, embed_prompt, system_prompt, system_embed_prompt)
        self.gw = gemma_wrapper

    def recommend(self, seed_ids: List[str], candidate_ids: List[str], use_genres=False, use_wiki=False, embed_seeds=False, contrast_num=0) -> List[str]:
        print('Seeds: ')
        print(seed_ids)
        print('Candidates: ')
        print(candidate_ids)
        iids = list(range(len(candidate_ids)))
        candidate_dict = bidict.bidict({candidate_ids[i]: iids[i] for i in range(len(candidate_ids))})
        join_char = '\n' if not use_wiki else '\n\n'

        seed_names = join_char.join([self.construct_artist_string(_id, use_genres, use_wiki) for _id in seed_ids])
        if embed_seeds:
            seed_names = self.embed_seed_artists(seed_names, contrast_num)

        candidate_names = join_char.join([self.artists[_id]['name'] + ('' if not use_genres else f" ({', '.join(self.artists[_id]['genres'])})") for _id in candidate_ids])
        candidate_dict_text = '\n'.join([f"{candidate_dict[_id]}: {self.artists[_id]['name']}" for _id in candidate_ids])
        _prompt = self.prompt.format(seeds=seed_names, candidates=candidate_names, candidate_key=candidate_dict_text)
        print(_prompt)

        text = self.gw.get_response(_prompt)
        print("Response Text: ")
        print(text, end="\n\n")
        return self.parse_output(text, candidate_dict)

    def embed_seed_artists(self, seed_string, contrast_num=0):
        _prompt = None
        if contrast_num:
            contrasting_seeds = sample(self.seeds, contrast_num)
            other_seeds_string = ""
            for i, user_seeds in enumerate(contrasting_seeds):
                other_seeds_string += f"User {i+1}\n"
                for seed_id in user_seeds:
                    other_seeds_string += f"- {self.artists[seed_id]['name']}\n"
                other_seeds_string += '\n'
            _prompt = self.embed_prompt.format(seeds=seed_string, other_seeds=other_seeds_string.strip())
        else:
            _prompt = self.embed_prompt.format(seeds=seed_string)

        print("Embed Prompt: ")
        print(_prompt, end="\n\n")

        text = self.gw.get_response(_prompt, self.system_embed_prompt)
        return text
