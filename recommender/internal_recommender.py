from typing import List

from recommender.base_recommender import Recommender


class InternalRecommender(Recommender):

    def __init__(self, artists, prompt, gemma_wrapper):
        super().__init__(artists, prompt)
        self.gw = gemma_wrapper

    def recommend(self, seed_ids: List[str], candidate_ids: List[str], use_genres=False, use_wiki=False, embed_seeds=False, embed_contrast=False) -> List[str]:
        #print('Seeds: ')
        print(seed_ids)
        #print('Candidates: ')
        print(candidate_ids)
        iids = list(range(len(candidate_ids)))
        join_char = '\n' if not use_wiki else '\n\n'
        candidate_dict = bidict.bidict({candidate_ids[i]: iids[i] for i in range(len(candidate_ids))})
        seed_names = join_char.join([self.construct_artist_string(_id, use_genres, use_wiki) for _id in seed_ids])
        candidate_names = join_char.join([self.artists[_id]['name'] + ('' if not use_genres else f" ({', '.join(self.artists[_id]['genres'])})") for _id in candidate_ids])
        candidate_dict_text = '\n'.join([f"{candidate_dict[_id]}: {self.artists[_id]['name']}" for _id in candidate_ids])
        _prompt = self.prompt.format(seeds=seed_names, candidates=candidate_names, candidate_key=candidate_dict_text)
        text = self.gw.get_response(_prompt)
        print("Response Text: ")
        print(text, end="\n\n")
        return self.parse_output(text, candidate_dict)
