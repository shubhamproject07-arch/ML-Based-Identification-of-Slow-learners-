import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

np.random.seed(42)
n_samples = 400

# Feature definitions
attendance = np.random.uniform(50, 100, n_samples)
internal_test_1 = np.random.uniform(10, 50, n_samples)
internal_test_2 = np.random.uniform(10, 50, n_samples)
assignment_score = np.random.uniform(40, 100, n_samples)
study_hours_weekly = np.random.uniform(2, 25, n_samples)

# Logic: Composite metric determines whether extra academic support is needed
composite_score = (
    0.30 * attendance
    + 0.25 * (internal_test_1 * 2)
    + 0.25 * (internal_test_2 * 2)
    + 0.10 * assignment_score
    + 0.10 * (study_hours_weekly * 4)
)

# Target: 1 = Needs Support / Slow Learner, 0 = Regular / Advanced Learner
labels = np.where(composite_score < 62, 1, 0)

df = pd.DataFrame(
    {
        "attendance": attendance,
        "internal_test_1": internal_test_1,
        "internal_test_2": internal_test_2,
        "assignment_score": assignment_score,
        "study_hours_weekly": study_hours_weekly,
        "needs_support": labels,
    }
)

os.makedirs("data", exist_ok=True)
df.to_csv("data/student_data.csv", index=False)

# Train/Test Split & Fit
X = df.drop(columns=["needs_support"])
y = df["needs_support"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=100, max_depth=6, random_state=42, class_weight="balanced"
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save artifacts
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/random_forest_model.pkl")
print("Model exported to model/random_forest_model.pkl")