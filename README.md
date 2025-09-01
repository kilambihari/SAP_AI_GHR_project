
# SAP GHR + AI: Employee Performance & Career Path Prediction

This is a ready-to-run demo that predicts employee performance, suggests likely next roles (career path) using a Markov chain, and recommends trainings based on the current role.

## Quickstart (Local)
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Train models
python src/train_performance_model.py
python src/train_career_markov.py
python src/recommend_training.py

# Run the app
streamlit run src/app_streamlit.py
```

## Files
- `data/hr_sample.csv` — synthetic HR dataset for demo
- `src/train_performance_model.py` — trains a RandomForest classifier
- `src/train_career_markov.py` — builds Markov transition probabilities from role sequences
- `src/recommend_training.py` — saves a simple role→training mapping
- `src/utils.py` — helper functions
- `src/app_streamlit.py` — demo Streamlit UI

## How to Connect to SAP GHR (Outline)
1. Use SAP BTP Integration Suite / OData APIs for SuccessFactors to extract:
   - Employees, Employment Info, Job Information, Performance Ratings, Learning History.
2. Replace `data/hr_sample.csv` with real merged tables (one row per employee with the features used by the model).
3. Retrain the models (scripts above) and redeploy.
4. Optionally containerize and deploy on SAP BTP, SAP AI Core, or any container platform.

## Notes
- This is a demo for learning. For production, add MLOps (versioning, monitoring), bias/fairness checks, consent & governance.
- Feature engineering, hyperparameter tuning, and richer sequence models (e.g., RNN) can improve accuracy.
- Ensure compliance with company data privacy policies.
