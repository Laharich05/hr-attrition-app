import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="HR Attrition Predictor", page_icon="👥", layout="wide")

@st.cache_resource
def load_artifacts():
    model    = joblib.load("hr_model.pkl")
    scaler   = joblib.load("hr_scaler.pkl")
    features = joblib.load("hr_features.pkl")
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

st.title("👥 HR Employee Attrition Predictor")
st.markdown("Predict whether an employee is likely to leave the organization.")
st.divider()

with st.sidebar:
    st.header("ℹ️ About")
    st.info("Dataset: IBM HR Analytics\nModel: Random Forest\nAccuracy: ~84%")

st.subheader("Enter Employee Details")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal**")
    age            = st.slider("Age", 18, 65, 35)
    gender         = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education      = st.selectbox("Education", [1,2,3,4,5],
                       format_func=lambda x:{1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])
    education_field = st.selectbox("Education Field", [
                        "Life Sciences","Medical","Marketing",
                        "Technical Degree","Human Resources","Other"])
    distance_home  = st.slider("Distance From Home (km)", 1, 30, 5)

with col2:
    st.markdown("**Job**")
    department     = st.selectbox("Department", ["Sales","Research & Development","Human Resources"])
    job_role       = st.selectbox("Job Role", [
                        "Sales Executive","Research Scientist","Laboratory Technician",
                        "Manufacturing Director","Healthcare Representative","Manager",
                        "Sales Representative","Research Director","Human Resources"])
    job_level      = st.selectbox("Job Level", [1,2,3,4,5])
    job_involvement = st.selectbox("Job Involvement", [1,2,3,4])
    job_satisfaction = st.selectbox("Job Satisfaction", [1,2,3,4])
    business_travel = st.selectbox("Business Travel", ["Non-Travel","Travel_Rarely","Travel_Frequently"])
    overtime       = st.selectbox("OverTime", ["Yes","No"])

with col3:
    st.markdown("**Compensation & Experience**")
    monthly_income      = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
    daily_rate          = st.number_input("Daily Rate", 100, 1500, 800)
    hourly_rate         = st.number_input("Hourly Rate", 30, 100, 65)
    monthly_rate        = st.number_input("Monthly Rate", 2000, 27000, 14000)
    percent_hike        = st.slider("Percent Salary Hike", 11, 25, 15)
    stock_option        = st.selectbox("Stock Option Level", [0,1,2,3])
    total_working_years = st.slider("Total Working Years", 0, 40, 10)
    num_companies       = st.slider("Num Companies Worked", 0, 9, 2)
    training_times      = st.slider("Training Times Last Year", 0, 6, 2)

col4, col5 = st.columns(2)
with col4:
    years_at_company   = st.slider("Years at Company", 0, 40, 5)
    years_in_role      = st.slider("Years in Current Role", 0, 20, 3)
    years_since_promo  = st.slider("Years Since Last Promotion", 0, 15, 2)
    years_with_manager = st.slider("Years With Current Manager", 0, 20, 3)
with col5:
    perf_rating        = st.selectbox("Performance Rating", [1,2,3,4])
    env_satisfaction   = st.selectbox("Environment Satisfaction", [1,2,3,4])
    relationship_sat   = st.selectbox("Relationship Satisfaction", [1,2,3,4])
    work_life_balance  = st.selectbox("Work Life Balance", [1,2,3,4])

st.divider()

if st.button("🔍 Predict Attrition", use_container_width=True, type="primary"):

    # Build dict matching EXACT feature names from training
    raw = {
        "Age":                      float(age),
        "DailyRate":                float(daily_rate),
        "DistanceFromHome":         float(distance_home),
        "Education":                float(education),
        "EnvironmentSatisfaction":  float(env_satisfaction),
        "Gender":                   1.0 if gender == "Male" else 0.0,
        "HourlyRate":               float(hourly_rate),
        "JobInvolvement":           float(job_involvement),
        "JobLevel":                 float(job_level),
        "JobSatisfaction":          float(job_satisfaction),
        "MonthlyIncome":            float(monthly_income),
        "MonthlyRate":              float(monthly_rate),
        "NumCompaniesWorked":       float(num_companies),
        "OverTime":                 1.0 if overtime == "Yes" else 0.0,
        "PercentSalaryHike":        float(percent_hike),
        "PerformanceRating":        float(perf_rating),
        "RelationshipSatisfaction": float(relationship_sat),
        "StockOptionLevel":         float(stock_option),
        "TotalWorkingYears":        float(total_working_years),
        "TrainingTimesLastYear":    float(training_times),
        "WorkLifeBalance":          float(work_life_balance),
        "YearsAtCompany":           float(years_at_company),
        "YearsInCurrentRole":       float(years_in_role),
        "YearsSinceLastPromotion":  float(years_since_promo),
        "YearsWithCurrManager":     float(years_with_manager),
        # One-hot encoded columns (drop_first=True)
        "BusinessTravel_Travel_Frequently": 1.0 if business_travel == "Travel_Frequently" else 0.0,
        "BusinessTravel_Travel_Rarely":     1.0 if business_travel == "Travel_Rarely" else 0.0,
        "Department_Research & Development": 1.0 if department == "Research & Development" else 0.0,
        "Department_Sales":                  1.0 if department == "Sales" else 0.0,
        "EducationField_Life Sciences":      1.0 if education_field == "Life Sciences" else 0.0,
        "EducationField_Marketing":          1.0 if education_field == "Marketing" else 0.0,
        "EducationField_Medical":            1.0 if education_field == "Medical" else 0.0,
        "EducationField_Other":              1.0 if education_field == "Other" else 0.0,
        "EducationField_Technical Degree":   1.0 if education_field == "Technical Degree" else 0.0,
        "JobRole_Human Resources":           1.0 if job_role == "Human Resources" else 0.0,
        "JobRole_Laboratory Technician":     1.0 if job_role == "Laboratory Technician" else 0.0,
        "JobRole_Manager":                   1.0 if job_role == "Manager" else 0.0,
        "JobRole_Manufacturing Director":    1.0 if job_role == "Manufacturing Director" else 0.0,
        "JobRole_Research Director":         1.0 if job_role == "Research Director" else 0.0,
        "JobRole_Research Scientist":        1.0 if job_role == "Research Scientist" else 0.0,
        "JobRole_Sales Executive":           1.0 if job_role == "Sales Executive" else 0.0,
        "JobRole_Sales Representative":      1.0 if job_role == "Sales Representative" else 0.0,
        "MaritalStatus_Married":             1.0 if marital_status == "Married" else 0.0,
        "MaritalStatus_Single":              1.0 if marital_status == "Single" else 0.0,
    }

    # CRITICAL: exact column order from training
    X_input = pd.DataFrame([raw]).reindex(columns=feature_names, fill_value=0).astype(float)
    X_scaled = scaler.transform(X_input)

    pred       = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if pred == 1:
            st.error("⚠️ HIGH ATTRITION RISK — This employee is likely to leave!")
        else:
            st.success("✅ LOW ATTRITION RISK — This employee is likely to stay!")
    with col_r2:
        st.metric("Attrition Probability", f"{pred_proba[1]*100:.1f}%")
        st.metric("Retention Probability", f"{pred_proba[0]*100:.1f}%")

    st.subheader("Probability Breakdown")
    st.bar_chart(pd.DataFrame({
        "Outcome": ["Will Stay", "Will Leave"],
        "Probability": [pred_proba[0], pred_proba[1]]
    }).set_index("Outcome"))
