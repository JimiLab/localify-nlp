import json


with open('experiments/Score Tracker.json') as file:
    a = json.load(file)
    for key, value in a.items():
        for v in value:
            print(key, len(v))


with open('nlp_seeds_anonymized.json') as file:
    a = json.load(file)
    print(len(a))
