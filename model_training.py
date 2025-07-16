import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
df = pd.read_csv("employee_salary_prediction.csv")

# Clean missing or invalid data
df = df.dropna(subset=["Salary"])
df = df.dropna()

# Encode categorical columns
le_gender = LabelEncoder()
le_edu = LabelEncoder()
le_job = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Education Level"] = le_edu.fit_transform(df["Education Level"])
df["Job Title"] = le_job.fit_transform(df["Job Title"])

# Prepare features and label
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Continue with train-test split, model, etc...
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

# Save model and encoders
with open("employee_salary_prediction.pkl", "wb") as f:
    pickle.dump((model, le_gender, le_edu, le_job), f)
