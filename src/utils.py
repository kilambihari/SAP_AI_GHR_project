
import json, joblib

def load_performance_model(path="models/performance_model.joblib"):
    return joblib.load(path)

def load_markov(path="models/markov_transitions.json"):
    with open(path) as f:
        return json.load(f)

def next_role_candidates(current_role, transitions, top_k=3):
    probs = transitions.get(current_role, {})
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]

def load_training_catalog(path="models/training_catalog.json"):
    with open(path) as f:
        return json.load(f)

def trainings_for_role(current_role, catalog, top_k=3):
    items = catalog.get(current_role, [])
    return items[:top_k]
