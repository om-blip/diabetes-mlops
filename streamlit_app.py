import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("models/model.pkl")

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("🩺 Diabetes Risk Prediction System")
st.markdown(
    """
This application predicts **diabetes risk** using clinical and lifestyle data  
based on the **CDC Behavioral Risk Factor dataset**.
"""
)

st.divider()

# =========================
# INPUT SECTION
# =========================

st.subheader("🧍 Personal & Health Information")

HighBP = st.selectbox(
    "High Blood Pressure",
    [0, 1],
    help="0 = No, 1 = Yes"
)

HighChol = st.selectbox(
    "High Cholesterol",
    [0, 1],
    help="0 = No, 1 = Yes"
)

CholCheck = st.selectbox(
    "Cholesterol Checked in Last 5 Years",
    [0, 1],
    help="0 = No, 1 = Yes"
)

BMI = st.number_input(
    "Body Mass Index (BMI)",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    help="BMI = weight (kg) / height² (m²)"
)

Smoker = st.selectbox(
    "Smoked at least 100 cigarettes in lifetime",
    [0, 1],
    help="0 = No, 1 = Yes"
)

Stroke = st.selectbox(
    "Ever had a stroke",
    [0, 1],
    help="0 = No, 1 = Yes"
)

HeartDisease = st.selectbox(
    "Heart disease or heart attack",
    [0, 1],
    help="0 = No, 1 = Yes"
)

PhysActivity = st.selectbox(
    "Physical activity in last 30 days",
    [0, 1],
    help="0 = No, 1 = Yes"
)

Fruits = st.selectbox(
    "Consumes fruits daily",
    [0, 1],
    help="0 = No, 1 = Yes"
)

Veggies = st.selectbox(
    "Consumes vegetables daily",
    [0, 1],
    help="0 = No, 1 = Yes"
)

HvyAlcohol = st.selectbox(
    "Heavy alcohol consumption",
    [0, 1],
    help="0 = No, 1 = Yes"
)

AnyHealthcare = st.selectbox(
    "Has health insurance",
    [0, 1],
    help="0 = No, 1 = Yes"
)

NoDocbcCost = st.selectbox(
    "Could not see doctor due to cost",
    [0, 1],
    help="0 = No, 1 = Yes"
)

GenHlth = st.slider(
    "General Health",
    1, 5,
    help="1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor"
)

MentHlth = st.slider(
    "Mental health bad days (last 30 days)",
    0, 30
)

PhysHlth = st.slider(
    "Physical health bad days (last 30 days)",
    0, 30
)

DiffWalk = st.selectbox(
    "Difficulty walking or climbing stairs",
    [0, 1],
    help="0 = No, 1 = Yes"
)

Sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

Age = st.slider(
    "Age Group",
    1, 13,
    help="""
    1: 18–24  
    2: 25–29  
    3: 30–34  
    4: 35–39  
    5: 40–44  
    6: 45–49  
    7: 50–54  
    8: 55–59  
    9: 60–64  
    10: 65–69  
    11: 70–74  
    12: 75–79  
    13: 80+
    """
)

Education = st.slider(
    "Education Level",
    1, 6,
    help="""
    1: Never attended school  
    2: Elementary  
    3: Some high school  
    4: High school graduate  
    5: Some college  
    6: College graduate
    """
)

Income = st.slider(
    "Income Level",
    1, 8,
    help="""
    1: < $10k  
    2: $10k–15k  
    3: $15k–20k  
    4: $20k–25k  
    5: $25k–35k  
    6: $35k–50k  
    7: $50k–75k  
    8: > $75k
    """
)

# Convert gender
Sex = 1 if Sex == "Male" else 0

# =========================
# PREDICTION
# =========================

if st.button("🔮 Predict Diabetes Risk"):
    features = np.array([[
        HighBP, HighChol, CholCheck, BMI, Smoker,
        Stroke, HeartDisease, PhysActivity,
        Fruits, Veggies, HvyAlcohol,
        AnyHealthcare, NoDocbcCost,
        GenHlth, MentHlth, PhysHlth,
        DiffWalk, Sex, Age, Education, Income
    ]])

    prediction = model.predict(features)[0]

    st.divider()

    if prediction == 0:
        st.success("✅ No Diabetes Detected")
    elif prediction == 1:
        st.warning("⚠️ Prediabetes Detected")
    else:
        st.error("🚨 Diabetes Detected — Please consult a doctor")
