import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained Random Forest model
model = joblib.load('randomforest.pkl')

st.write("""
# Diabetes Prediction App
This app predicts the **likelihood of diabetes** based on user input!
""")

st.sidebar.header('User Input Parameters')

# Helper function to map Yes/No input to 1/0
def yes_no_mapping(label):
    return st.sidebar.selectbox(label, options=["Yes", "No"], index=1)

def user_input_features():
    high_blood_pressure = 1 if yes_no_mapping('High blood pressure') == "Yes" else 0
    high_cholesterol = 1 if yes_no_mapping('High cholesterol') == "Yes" else 0
    cholesterol_check = 1 if yes_no_mapping('Cholesterol check (past 5 years)') == "Yes" else 0
    bmi = st.sidebar.slider('BMI', 10, 98, 25)
    smoker = 1 if yes_no_mapping('Smoker') == "Yes" else 0
    stroke = 1 if yes_no_mapping('Stroke history') == "Yes" else 0
    heart_disease = 1 if yes_no_mapping('Heart disease') == "Yes" else 0
    physical_activity = 1 if yes_no_mapping('Physical activity (past 30 days)') == "Yes" else 0
    fruits = 1 if yes_no_mapping('Consumes fruits daily') == "Yes" else 0
    veggies = 1 if yes_no_mapping('Consumes veggies daily') == "Yes" else 0
    heavy_alcohol_consumer = 1 if yes_no_mapping('Heavy alcohol consumer') == "Yes" else 0
    any_health_care = 1 if yes_no_mapping('Has health care') == "Yes" else 0
    no_doctor_cost = 1 if yes_no_mapping('Could not see doctor due to cost') == "Yes" else 0
    general_health = st.sidebar.slider('General health (1=Excellent, 5=Poor)', 1, 5, 3)
    mental_health = st.sidebar.slider('Mental health (days of issues)', 0, 30, 5)
    physical_health = st.sidebar.slider('Physical health (days of issues)', 0, 30, 5)
    difficulty_walking = 1 if yes_no_mapping('Difficulty walking') == "Yes" else 0
    sex = 1 if st.sidebar.selectbox('Sex', options=["Female", "Male"]) == "Male" else 0
    age = st.sidebar.slider('Age group (1=18-24, 13=80+)', 1, 13, 5)
    education = st.sidebar.slider('Education level (1=No schooling, 6=Postgraduate)', 1, 6, 4)
    income = st.sidebar.slider('Income level (1=<$10,000, 8=>$75,000)', 1, 8, 4)

    data = {
        'HighBP': high_blood_pressure,
        'HighChol': high_cholesterol,
        'CholCheck': cholesterol_check,
        'BMI': bmi,
        'Smoker': smoker,
        'Stroke': stroke,
        'HeartDiseaseorAttack': heart_disease,
        'PhysActivity': physical_activity,
        'Fruits': fruits,
        'Veggies': veggies,
        'HvyAlcoholConsump': heavy_alcohol_consumer,
        'AnyHealthcare': any_health_care,
        'NoDocbcCost': no_doctor_cost,
        'GenHlth': general_health,
        'MentHlth': mental_health,
        'PhysHlth': physical_health,
        'DiffWalk': difficulty_walking,
        'Sex': sex,
        'Age': age,
        'Education': education,
        'Income': income
    }

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

# Make predictions
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Prediction')
# Mapping prediction output to readable labels
prediction_labels = {0: 'No', 1: 'Pre-diabetes', 2: 'Yes'}
st.write('Diabetes Likelihood:', prediction_labels[prediction[0]])

st.subheader('Prediction Probability')
st.write(prediction_proba)
