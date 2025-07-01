from typing import List

from recommender.base_recommender import Recommender


class PopularityRecommender(Recommender):

    def __init__(self, artists, prompt=None):
        super().__init__(artists, prompt)

    def recommend(self, seed_ids: List[str], candidate_ids: List[str], use_genres=False, use_wiki=False, embed_seeds=False, embed_contrast=False) -> List[str]:
        return sorted(candidate_ids, key=lambda _id: self.artists[_id]['spotify_popularity'], reverse=True)
