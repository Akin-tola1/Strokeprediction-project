import pandas as pd 
import numpy as np 
import seaborn as sns 
import streamlit as st 
import matplotlib.pyplot as plt 
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = data.copy()

scaler = StandardScaler()
encoder = LabelEncoder()

# # copy your data
# new_data = data.copy()
# num= new_data.select_dtypes(include = 'number')
# cat= new_data.select_dtypes(exclude = 'number')

encoded = {}
# Encode the categorical data set
for i in df.select_dtypes(exclude = 'number').columns:
    encoder = LabelEncoder()
    df[i] = encoder.fit_transform(df[i])
    encoded[i] = encoder


st.markdown("<h1 style = 'color: #7F27FF; text-align: center; font-family: helvetica '>STROKE PREDICTION OCCURENCE</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #12372A; text-align: center; font-family: cursive '>Built By DEJI</h4>", unsafe_allow_html = True)

st.image('pngwing.com (11).png',width= 20, use_column_width = True)

st.markdown("<br>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html = True)

st.markdown("<p>The Stroke Prediction Ocurrence using Machine Learning project aims toThis project aims to predict stroke occurrence based on demographic, lifestyle, and health-related factors. By analyzing data from medical records and health surveys, the model identifies individuals at risk of stroke, facilitating early intervention and prevention strategies. Through rigorous data preprocessing, feature engineering, and model training, the project delivers a reliable predictive tool for healthcare practitioners and policymakers to improve stroke management and public health outcomes..</p>", unsafe_allow_html= True)

st.markdown("<br>", unsafe_allow_html = True)
st.dataframe(data, use_container_width= True)


st.sidebar.image('pngwing.com (10).png',caption = 'Welcome Dear User')
st.sidebar.write('Feature Input')
gender = st.sidebar.selectbox('Gender', data.gender.unique())
age = st.sidebar.number_input('age', df['age'].min(), df['age'].max())
hypertension = st.sidebar.number_input('hypertensive', df['hypertension'].min(), df['hypertension'].max())
heart_disease=st.sidebar.number_input('heart_disease', df['heart_disease'].min(), df['heart_disease'].max())
ever_married = st.sidebar.selectbox('Marital Status', data.ever_married.unique())
work_type = st.sidebar.selectbox('Occupation', data.work_type.unique())
Residence_type = st.sidebar.selectbox('Type of Residence', data.Residence_type.unique())
avg_glucose_level=st.sidebar.number_input('avg_glucose_level', df['avg_glucose_level'].min(), df['avg_glucose_level'].max())
bmi=st.sidebar.number_input('bmi', df['bmi'].min(), df['bmi'].max())
smoking_status = st.sidebar.selectbox('Smoking Status', data.smoking_status.unique())

# encoded columns and refit

st.markdown("<br>", unsafe_allow_html= True)
st.write('Input Variables')
input_var = pd.DataFrame({'gender':[gender], 'age':[age], 'hypertension':[hypertension], 'heart_disease':[heart_disease], 'ever_married':[ever_married],'work_type':[work_type], 'Residence_type':[Residence_type], 'avg_glucose_level':[avg_glucose_level], 'bmi':[bmi],'smoking_status':[smoking_status]})

input_var['gender'] = encoded['gender'].transform(input_var['gender'])
input_var['ever_married'] = encoded['ever_married'].transform(input_var['ever_married'])
input_var['work_type'] = encoded['work_type'].transform(input_var['work_type'])
input_var['Residence_type'] = encoded['Residence_type'].transform(input_var['Residence_type'])
input_var['smoking_status'] = encoded['smoking_status'].transform(input_var['smoking_status'])


st.dataframe(input_var)


model = joblib.load('stroke.pkl')

predicter = st.button('Stroke Prediction')
if predicter:
    prediction = model.predict(input_var)
    st.success(f'The Predicted value for your Stroke Prediction is {prediction}')
    
