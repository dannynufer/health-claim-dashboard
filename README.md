# 🏥 Health Insurance Claim Predictor

---

## 📚 Overview

This health insurance claim predictor was built in Python, utilizing the `pandas`, `numpy`, `statsmodels`, and `streamlit` libraries.  
It uses a Generalized Linear Model (GLM) trained on over 1,300 health claim observations to predict the size of a claim based on:

- Age
- Gender
- Smoking status
- Number of children
- Diabetic status
- Blood pressure
- BMI

The project features a clean Streamlit dashboard, allowing users to easily input different values for each explanatory variable and instantly receive a prediction.

👉 [**Live Dashboard 🚀**](https://health-claim-dashboard.streamlit.app/)

---

## 🚀 Features

- GLM (Gamma) model trained on real-world insurance data
- Clean, user-friendly Streamlit web app
- Dynamic claim amount prediction based on user inputs

---

## ⚙️ How to Run Locally

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/claims-risk-analyzer.git
    cd claims-risk-analyzer
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Launch the Streamlit app:
    ```bash
    streamlit run app/health_claim_dashboard.py
    ```

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Statsmodels
- Streamlit
- Matplotlib

---

## 📣 Acknowledgments

- **Dataset source:** Credit to Sumit Kumar Shukla for the publicly available insurance sample data.

---

## 🧠 About

Predicting health insurance claim amounts using a Generalized Linear Model (Gamma distribution).  
Includes a clean Streamlit dashboard for user interaction.

---
