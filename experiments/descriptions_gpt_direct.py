from openai import OpenAI
import datetime

from evaluator import Evaluator
from recommender.gpt_recommender import GPTRecommender
from utils import load_json, load_yaml


def main():
    name = "GPT Descriptions"

    artists = load_json("../nlp_artists_filtered.json")
    seeds = load_json("../nlp_seeds_anonymized.json")
    artist_ids = [k for k in artists.keys()]

    conf = load_yaml("../conf.yaml")
    client = OpenAI(api_key=conf['openai-key'])

    prompt = """You are an expert in music recommendation. Your specialty is in ranking a list of artists by how similar each one is to a different set of artists that someone already knows.
You are presented with a client who frequently listens to the following artists:
{seeds}
You are asked to use your expert knowledge of these artists to rank the following artists (on which you are also an expert) in order from most recommended to least recommended:
{candidates}
You must present these recommendations in a very specific way. Each candidate artist that you are recommending has an integer ID associated with them. The key is as follows:
{candidate_key}
You must finish your response by listing your recommendations only using these ids, separated by commas, and surrounded by <>.
An example recommendation would be as follows:
<#,#,#,#,#,#,#,#,#,#,...>
With each hashtag replaced by the artist you recommend in that position. You must abide by the following rules:
- Rank all of the artists in the candidate set I provided you, not just however many are in my example recommendation
- Do not include any artists not in that candidate set
- Your response must be formatted as I have said, in a list of comma separated ids (governed by the key I provided) and surrounded by angle brackets/chevrons (<>)
"""

    gpt = GPTRecommender(artists, prompt, client)
    evaluator = Evaluator(gpt, artist_ids, seeds)
    results = evaluator.eval_model(
        result_path=name+' '+str(datetime.datetime.now())+".txt",
        experiment_name=name,
    )
    print(f"{name} ChatGPT Score:", results.mean)


if __name__ == "__main__":
    main()
