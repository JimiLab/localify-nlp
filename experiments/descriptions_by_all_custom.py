from openai import OpenAI
import datetime

from evaluator import Evaluator
from recommender.gpt_recommender import GPTRecommender
from utils import load_json, load_yaml


def main():
    name = "Descriptions By All Custom Example"

    artists = load_json("../nlp_artists_filtered.json")
    seeds = load_json("../nlp_seeds_anonymized.json")
    artist_ids = [k for k in artists.keys()]

    conf = load_yaml("../conf.yaml")
    client = OpenAI(api_key=conf['openai-key'])

    prompt = """You are an expert in music recommendation. Your specialty is in ranking a list of artists based on a textual listening profile that you are provided.
You are presented with a client with the following listening profile:

\"\"\"
{seeds}
\"\"\"

You are asked to use your expert knowledge of musical artists and your complete understanding of the listening profile to rank the following artists (on which you are also an expert) in order from most recommended to least recommended:
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

    embed_prompt = """You are an expert in describing people's music listening habits. You are presented with a client who listens to the following artists:
{seeds}

You are also presented with the following other users' familiar artists:
{other_seeds}

Give a textual description of this person's listening habits, without using artist names. Focus on how your client's listening habits differ from the habits of the other users (what makes them unique). Do not directly mention these other users, and do not pander to the client. Give an accurate description that gives the best possible summary of the artists that they like.
The description should be designed such that a third party could use it to make artist recommendations, and it should follow the following example:

\"\"\"
My music taste is a mix of pop, musical theatre, r&b and dance/electronic. I used to be in theatre, so I love a great singer, and I really value the quality of voice and the skill of the band. As such, I love when artists use real instruments and studio musicians. 
I do also like music at the other side of the spectrum that's electronic, but not synth trying to be instruments. 
I like upbeat music, so I don't usually listen to sad songs but I will if the vocals are amazing, and I love a dreamy relaxing r&b track. 
I also love a rap feature on a pop song.
\"\"\"

as a structure and guide, but the details should pertain to the client it is written for.
The description should be written in first-person, from the perspective of the client.
"""

    gpt = GPTRecommender(artists, prompt, client, embed_prompt=embed_prompt, seeds=seeds)
    evaluator = Evaluator(gpt, artist_ids, seeds)
    results = evaluator.eval_model(
        result_path=name+' '+str(datetime.datetime.now())+".txt",
        experiment_name=name,
        embed_seeds=True,
        contrast_num=5
    )
    print(f"{name} ChatGPT Score:", results.mean)


if __name__ == "__main__":
    main()
