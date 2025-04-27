
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# Set the page config
st.set_page_config(page_title="Health Insurance Claim Predictor", layout="wide")

# Page title
st.title("🏥 Health Insurance Claim Predictor")

st.markdown(
    """
    Predict expected insurance claim amounts based on patient demographics and health metrics.
    Fill out the information on the left and click **Predict**!
    """
)

# Sidebar for inputs
st.sidebar.header("👤 Demographic Information")

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
gender_male = st.sidebar.selectbox("Gender", options=["Male", "Female"])
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=0)

st.sidebar.header("🩺 Health Metrics")

bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
bloodpressure = st.sidebar.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
smoker_Yes = st.sidebar.selectbox("Smoker", options=["Yes", "No"])
diabetic_Yes = st.sidebar.selectbox("Diabetic", options=["Yes", "No"])


# Load model coefficients
def load_model():
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, '..', 'data', 'insurance_data.csv')
    file_path = os.path.abspath(file_path)

    df = pd.read_csv(file_path)
    df = df.dropna(subset=['age'])
    df_encoded = pd.get_dummies(df, columns=['gender', 'smoker', 'diabetic'], drop_first=True)
    
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
    
    X = df_encoded[['age', 'bmi', 'bloodpressure', 'children', 'gender_male', 'smoker_Yes', 'diabetic_Yes']]
    y = df_encoded['claim']
    X = sm.add_constant(X)
    
    model = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.log()))
    result = model.fit()
    
    return result

model = load_model()

# Prepare input for prediction
def prepare_input(age, bmi, bloodpressure, children, gender_male, smoker_Yes, diabetic_Yes):
    data = {
        'const': 1.0,
        'age': age,
        'bmi': bmi,
        'bloodpressure': bloodpressure,
        'children': children,
        'gender_male': 1 if gender_male == "Male" else 0,
        'smoker_Yes': 1 if smoker_Yes == "Yes" else 0,
        'diabetic_Yes': 1 if diabetic_Yes == "Yes" else 0
    }
    return pd.DataFrame([data])

if st.sidebar.button("Predict Claim Amount"):
    input_data = prepare_input(age, bmi, bloodpressure, children, gender_male, smoker_Yes, diabetic_Yes)
    predicted_claim = model.predict(input_data)[0]
    
    st.markdown(
        f"<h2 style='color: green;'>💰 Expected Claim Amount: £{predicted_claim:,.2f}</h2>", 
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)  

    st.markdown(
        f"<h4 style='color: white;'>Notes:</h4>", 
        unsafe_allow_html=True
    )
    
    if smoker_Yes == "Yes":
        st.info("🚬 Smoking is associated with significantly higher claim amounts.*")
    else:
        st.info("🚭 Non-smokers tend to have lower expected claim amounts.*")
        
    if gender_male == "Female":
        st.info("👩 Females claim slightly higher amounts on average.*")
    else:
        st.info("🧔‍♂️ Males claim slightly lower amounts on average.*")
    
    st.markdown(
        "<p style='font-size: 13px;'>* Based on historical data used to train the model.</p>", 
        unsafe_allow_html=True
    )


# --- Footer ---

st.markdown("---")  

st.markdown(
    """
    <p style='text-align: center; font-size: 17px;'>
    Created by <strong>Danny Nufer</strong> | 
    <a href='https://github.com/dannynufer/health-claim-dashboard' target='_blank'>View on GitHub</a>
    </p>
    """,
    unsafe_allow_html=True
)


st.markdown(
    "<p style='text-align: center; font-size: 12px;'>This tool is for educational and demonstration purposes only. Predictions are based on historical sample data and are not intended for commercial use.</p>", 
    unsafe_allow_html=True
)