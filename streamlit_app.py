import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
with open("employee_salary_prediction.pkl", "rb") as f:
    model, le_gender, le_edu, le_job = pickle.load(f)

st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered")

# App banner
st.image("assets/banner.png", use_container_width=True)

st.title("💼 Employee Salary Prediction")
st.markdown("### Fill the employee details below:")

# Input form
with st.form("salary_form"):
    age = st.slider("Age", 18, 65)
    gender = st.selectbox("Gender", le_gender.classes_)
    education = st.selectbox("Education Level", le_edu.classes_)
    job_title = st.selectbox("Job Title", le_job.classes_)
    experience = st.slider("Years of Experience", 0, 25)

    submit = st.form_submit_button("Predict Salary 💰")

if submit:
    # Encode inputs
    gender_encoded = le_gender.transform([gender])[0]
    edu_encoded = le_edu.transform([education])[0]
    job_encoded = le_job.transform([job_title])[0]

    input_df = pd.DataFrame([[age, gender_encoded, edu_encoded, job_encoded, experience]],
                            columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"])

    prediction = model.predict(input_df)[0]
    st.success(f"🧾 Estimated Salary: ₹ {round(prediction, 2)}")
