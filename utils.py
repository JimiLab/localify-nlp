import json
import yaml


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


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
