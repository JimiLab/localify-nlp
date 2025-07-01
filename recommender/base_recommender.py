# This is an interface with three implementations;
# GPTRecommender which implements everything by making API calls to ChatGPT's API
# InternalRecommender which implements everything by generating responses from an internal model
# - For this experiment, we will use one of Google's open-source Gemma models
# PopularityRecommender which implements everything using only spotify popularity
from typing import List


class Recommender:

    def __init__(self, artists, prompt, seeds=None, embed_prompt=None):
        # essentially json of {artist_id: artist_data (genres tags, wiki, etc)}
        self.artists = artists
        self.prompt = prompt
        self.embed_prompt = embed_prompt
        # a list of lists of UUIDs representing artists
        self.seeds = seeds

    def recommend(self, seed_ids: List[str], candidate_ids: List[str], use_genres=False, use_wiki=False, embed_seeds=False, embed_contrast=False) -> List[str]:
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

    def construct_artist_string(self, _id, use_genres=False, use_wiki=False):
        if use_genres:
            return self.artists[_id]['name'] + f" ({', '.join(self.artists[_id]['genres'])})"
        if use_wiki:
            wiki_clipped = ' - ' + '. '.join(self.artists[_id]['wiki'].split('. ')[:5])
            return self.artists[_id]['name'] + wiki_clipped
        return self.artists[_id]['name']

    def embed_seed_artists(self, seed_string, embed_contrast=False):
        raise NotImplementedError(f'{self.__class__} class is an interface and not intended for instantiation.')
