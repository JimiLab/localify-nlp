import datetime

from evaluator import Evaluator
from recommender.internal_recommender import InternalRecommender
from recommender.wrappers.gemma_wrapper import GemmaWrapper
from utils import load_json
import yaml
from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch
import gc


def main():
    torch.cuda.empty_cache()
    gc.collect()

    artists = load_json("nlp_artists_filtered.json")
    seeds = load_json("nlp_seeds_anonymized.json")
    artist_ids = [k for k in artists.keys()]

    with open('conf.yaml') as file:
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

    prompt = """You are presented with a client who frequently listens to the following artists:
{seeds}

You are asked to use your expert knowledge of these artists to rank the following artists (on which you are also an expert) in order from most recommended to least recommended:
{candidates}

You must present these recommendations in a very specific way. Each candidate artist that you are recommending has an integer ID associated with them. The key is as follows:
{candidate_key}

You must finish your response by listing your recommendations only using these ids, separated by commas, and surrounded by <>.
An example recommendation would be as follows:
<#,#,#,#,#,#,#,#,#,#,...>
With each hashtag replaced by the artist you recommend in that position.
Provide me with only the ranked list of integer IDs associated with the ranking, do not enumerate your thought process."""

    name = "Gemma Direct"

    gw = GemmaWrapper(gemma_model, gemma_tokenizer)
    gemma = InternalRecommender(artists, prompt, gw)
    evaluator = Evaluator(gemma, artist_ids, seeds)
    score = evaluator.eval_model(
        result_path=name+' '+str(datetime.datetime.now())+".txt",
        experiment_name=name
    )
    print(f"{name} Score:", score)


if __name__ == "__main__":
    main()
