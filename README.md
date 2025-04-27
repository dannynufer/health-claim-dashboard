# Health Insurance Claim Predictor
---

## Overview
This health insurance claim predictor was created in python, utilising the `pandas`, `numpy`, `statsmodels` and `streamlit` libraries. 
It uses a GLM model trained on over 1000 health claim observations to predict the size of a health claim based on: 

- Age
- Sex
- Whether they are a smoker
- Number of children
- Whether they are diabetic
- Blood pressure
- BMI

Using streamlit, I created an intuitive dashboard allowing the user to easily input different values for each explanatory variable, and instantly get an output.

---

## Features 

- GLM (Gamma) model trained on real insurance data
- Clean, user-friendly Streamlit web app
- Dynamic claim amount prediction based on user inputs (age, BMI, blood pressure, smoking status, etc.)

---

## How to use
[Visit the Live Dashboard here! 🚀](https://health-claim-dashboard.streamlit.app/)

---

## Tech Stack 

- Python
- Pandas
- NumPy
- Statsmodels
- Streamlit
- Matplotlib

---

### Acknowledgments:
Data sourced [here.](https://www.kaggle.com/datasets/thedevastator/insurance-claim-analysis-demographic-and-health?resource=download) Credit: Sumit Kumar Shukla
