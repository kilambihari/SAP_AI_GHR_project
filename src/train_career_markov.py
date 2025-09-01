
import pandas as pd
import numpy as np
import json
import os

DATA_PATH = os.environ.get("DATA_PATH", "data/hr_sample.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/markov_transitions.json")

df = pd.read_csv(DATA_PATH)

# Build transition counts from Role_History sequences
from collections import defaultdict, Counter

trans_counts = defaultdict(Counter)

def parse_path(s):
    return [x.strip() for x in str(s).split(">")]

for path in df["Role_History"].dropna():
    roles = parse_path(path)
    for i in range(len(roles)-1):
        a, b = roles[i], roles[i+1]
        trans_counts[a][b] += 1

# Convert to probabilities
transitions = {}
for a, cnt in trans_counts.items():
    total = sum(cnt.values())
    probs = {b: c/total for b, c in cnt.items()}
    transitions[a] = probs

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "w") as f:
    json.dump(transitions, f, indent=2)
print(f"Saved Markov transitions to {MODEL_PATH}")
