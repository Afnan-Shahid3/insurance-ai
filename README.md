# Insurance Claim Cost Prediction System (AI Project)

## Project Overview

This project is an **AI-based insurance claim cost prediction system** built as part of an **Intro to Artificial Intelligence course**.

The system uses machine learning to predict the expected cost of an insurance claim based on user-provided details such as driver information, vehicle type, region risk, weather conditions, and accident type.

It also provides a simple explanation of why a prediction was made, making the system more interpretable and user-friendly.

---

## Objectives

- Predict insurance claim cost using machine learning
- Classify risk level (Low / Medium / High)
- Provide simple explanation for predictions
- Build a user-friendly web interface using Streamlit
- Demonstrate end-to-end AI pipeline (data → model → UI)

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

## Features

- 📥 Input insurance claim details via UI
- 🤖 Predict claim cost using ML model
- ⚠️ Classify risk level
- 🧾 Explain prediction in simple language
- 📈 Easy-to-use web interface

---

## Machine Learning Approach

- Problem Type: Regression
- Model Used: XGBoost / Random Forest
- Data Handling: Label Encoding + Basic preprocessing
- Evaluation: Train/Test split

---

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/insurance-ai.git
cd insurance-ai
```
### 2. Create virtual enviornment
```bash
python -m venv venv
```
### 3. Activate Enviorment
```bash
venv\scripts\activate
```
### 4. Install dependencies
```bash
pip install -r requirements.txt
```
### 5. Run Streamlit app
```bash
streamlit run app.py
```

---

## Project Team

- Syed Afnan — 30224
- Faraz Ahmed — 30503

---

## Learning Outcomes
- Understanding supervised machine learning (regression)
- Working with real-world structured data
- Model training and evaluation workflow
- Building interactive AI applications
- Basics of explainable AI

