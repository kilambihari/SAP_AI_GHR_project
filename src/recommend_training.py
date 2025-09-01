
import pandas as pd
import json
import os
from collections import Counter

DATA_PATH = os.environ.get("DATA_PATH", "data/hr_sample.csv")
OUT_PATH = os.environ.get("OUT_PATH", "models/training_catalog.json")

# Minimal rule-based mapping of roles to helpful trainings
# In real projects, learn from historical "employee x completed training y" -> performance uplift.
role_to_trainings = {
    "Sales Associate": ["Negotiation Basics", "CRM Deep Dive", "Pitch Practice"],
    "Senior Sales Associate": ["Advanced Negotiation", "Account Management"],
    "Team Lead (Sales)": ["Leadership Essentials", "Coaching Sales Teams"],
    "Sales Manager": ["Strategic Sales Planning", "Forecasting & Pipeline"],
    "HR Executive": ["Recruitment Fundamentals", "HR Analytics 101"],
    "Senior HR Executive": ["Compensation & Benefits", "Workforce Planning"],
    "HRBP": ["Business Partnering", "Change Management"],
    "HR Manager": ["Designing HR Strategy", "Labor Law Overview"],
    "Junior Engineer": ["Python for Data", "Git & DevOps Basics"],
    "Engineer": ["System Design Basics", "Cloud Practitioner"],
    "Senior Engineer": ["Advanced Algorithms", "Microservices"],
    "Tech Lead": ["Leading Engineering Teams", "Architecture Patterns"],
    "Engineering Manager": ["People Management", "Roadmapping"],
    "Analyst": ["Advanced Excel & Reporting", "Finance 101"],
    "Senior Analyst": ["Financial Modeling", "Data Visualization"],
    "Team Lead (Finance)": ["Leadership Essentials", "Risk & Controls"],
    "Finance Manager": ["Strategic Finance", "Budgeting & Planning"],
    "Ops Associate": ["Lean Basics", "Process Mapping"],
    "Senior Ops Associate": ["Root Cause Analysis", "Inventory Control"],
    "Ops Supervisor": ["Shift Planning", "Safety & Compliance"],
    "Ops Manager": ["SCM Overview", "Operational Excellence"],
    "Marketing Associate": ["Content Marketing", "SEO Basics"],
    "Senior Marketing Associate": ["Brand Storytelling", "Campaign Analytics"],
    "Brand Lead": ["Creative Direction", "Market Research"],
    "Marketing Manager": ["Go-To-Market Strategy", "Media Planning"]
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(role_to_trainings, f, indent=2)
print(f"Saved training catalog to {OUT_PATH}")
