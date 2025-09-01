
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

DATA_PATH = os.environ.get("DATA_PATH", "data/hr_sample.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/performance_model.joblib")

df = pd.read_csv(DATA_PATH)

# Target: Next year's performance rating (proxy: Past_Performance_Rating shifted by some noise)
# For demo, we use Past_Performance_Rating as target.
y = df["Past_Performance_Rating"]
X = df.drop(columns=["Employee_ID","Role_History","Current_Role","Past_Performance_Rating"])

num_cols = ["Age","Years_at_Company","Trainings_Completed","Certifications","Salary_LPA","Promotion_Last_Year"]
cat_cols = ["Gender","Department","Education"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight=None)

pipe = Pipeline([("pre", pre), ("clf", clf)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(pipe, MODEL_PATH)
print(f"Saved performance model to {MODEL_PATH}")
