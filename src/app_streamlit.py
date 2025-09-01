
import streamlit as st
import pandas as pd
from utils import load_performance_model, load_markov, next_role_candidates, load_training_catalog, trainings_for_role

st.set_page_config(page_title="SAP GHR: Performance & Career Path AI", layout="wide")
st.title("SAP GHR: Employee Performance & Career Path Prediction")    

uploaded = st.file_uploader("Upload employee CSV (or use demo data)", type=["csv"])    
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("data/hr_sample.csv")

st.subheader("Preview Data")
st.dataframe(df.head(20))

# Load models
perf_model = load_performance_model()
transitions = load_markov()
catalog = load_training_catalog()

st.subheader("Predict Performance & Recommend Career Path")    
row = st.selectbox("Choose an Employee_ID", df["Employee_ID"].tolist())
rec = df[df["Employee_ID"]==row].iloc[0]

# Prepare features same as training
X = rec.drop(labels=["Employee_ID","Role_History","Current_Role","Past_Performance_Rating"]).to_frame().T
y_pred = perf_model.predict(X)[0]

st.markdown(f"**Predicted Performance (next cycle):** {y_pred}")

current_role = rec["Current_Role"]
st.markdown(f"**Current Role:** {current_role}")

next_roles = next_role_candidates(current_role, transitions, top_k=3)
if next_roles:
    st.markdown("**Likely Next Roles:**")
    for role, p in next_roles:
        st.write(f"- {role} (prob ~ {p:.2f})")
else:
    st.info("No transition data for this role in demo dataset.")

st.markdown("**Recommended Trainings:**")
for t in trainings_for_role(current_role, catalog, top_k=3):
    st.write(f"- {t}")

st.caption("Demo app. Connect to SAP GHR via APIs for real deployment.")
