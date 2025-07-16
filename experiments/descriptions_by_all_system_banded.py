from openai import OpenAI
import datetime

from evaluator import Evaluator
from recommender.gpt_recommender import GPTRecommender
from utils import load_json, load_yaml


def main():
    artists = load_json("../nlp_artists_filtered.json")
    seeds_all = load_json("../nlp_seeds_categorized.json")
    seeds_low, seeds_med, seeds_high = seeds_all["low"], seeds_all["med"], seeds_all["high"]
    artist_ids = [k for k in artists.keys()]

    conf = load_yaml("../conf.yaml")
    client = OpenAI(api_key=conf['openai-key'])

    prompt = """You are presented with a client with the following listening profile:

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

    embed_prompt = """You are presented with a client who listens to the following artists:
{seeds}

You are also presented with the following other users' familiar artists:
{other_seeds}

Give a textual description of this person's listening habits, without using artist names. Focus on how your client's listening habits differ from the habits of the other users (what makes them unique). Do not directly mention these other users, and do not pander to the client. Give an accurate description that gives the best possible summary of the artists that they like.
The description should be designed such that a third party could use it to make artist recommendations, and it should follow the following example:

\"\"\"
My musical taste weaves velvety jazz-infused neo-soul with driving, synth-heavy electronic grooves that effortlessly blend warmth and futurism. 
I gravitate toward smoky, upright-bass-led lounge numbers that evoke intimate club corners, alongside pulsating house tracks that ignite late-night dancefloors. 
I refresh my sets with experimental ambient textures and lo-fi hip-hop beats that layer nostalgic vinyl crackle over head-nodding rhythms, while occasional avant-garde free-jazz injections add daring dissonance. 
This balance of cozy soulfulness and boundary-pushing sonic exploration speaks to my love of music that soothes and stimulates in equal measure. 
The result is a listening profile rooted in sophistication and spontaneity, comfort and curiosity, offering a journey that feels both familiar and thrilling.
\"\"\"

as a structure and guide, but the details should pertain to the client it is written for.
The description should be written in first-person, from the perspective of the client.
"""

    system_prompt = """You are an expert in music recommendation. Your specialty is in ranking a list of artists based on a textual listening profile that you are provided.
You are able to perfectly rank these artists in order of preference, where preference is defined by how much a user matching your provided listening profile will enjoy that artist.
"""

    system_embed_prompt = """You are an expert in describing people's music listening habits. 
You are experienced in writing clear, concise, and descriptive summaries of people's listening habits based on a list of artists that they like.
"""

    name_low = "Descriptions By All System Prompt Low Popularity"
    gpt_low = GPTRecommender(
        artists, prompt, client,
        embed_prompt=embed_prompt,
        seeds=seeds_low,
        system_prompt=system_prompt,
        system_embed_prompt=system_embed_prompt
    )
    evaluator_low = Evaluator(gpt_low, artist_ids, seeds_low)
    results_low = evaluator_low.eval_model(
        result_path=name_low + ' ' + str(datetime.datetime.now()) + ".txt",
        experiment_name=name_low,
        embed_seeds=True,
        contrast_num=5
    )
    print(f"{name_low} ChatGPT Score:", results_low.mean)

    name_med = "Descriptions By All System Prompt Medium Popularity"
    gpt_med = GPTRecommender(
        artists, prompt, client,
        embed_prompt=embed_prompt,
        seeds=seeds_med,
        system_prompt=system_prompt,
        system_embed_prompt=system_embed_prompt
    )
    evaluator_med = Evaluator(gpt_med, artist_ids, seeds_med)
    results_med = evaluator_med.eval_model(
        result_path=name_med + ' ' + str(datetime.datetime.now()) + ".txt",
        experiment_name=name_med,
        embed_seeds=True,
        contrast_num=5
    )
    print(f"{name_med} ChatGPT Score:", results_med.mean)

    name_high = "Descriptions By All System Prompt High Popularity"
    gpt_high = GPTRecommender(
        artists, prompt, client,
        embed_prompt=embed_prompt,
        seeds=seeds_high,
        system_prompt=system_prompt,
        system_embed_prompt=system_embed_prompt
    )
    evaluator_high = Evaluator(gpt_high, artist_ids, seeds_high)
    results_high = evaluator_high.eval_model(
        result_path=name_high + ' ' + str(datetime.datetime.now()) + ".txt",
        experiment_name=name_high,
        embed_seeds=True,
        contrast_num=5
    )
    print(f"{name_high} ChatGPT Score:", results_high.mean)


if __name__ == "__main__":
    main()
