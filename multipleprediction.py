# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 21:35:01 2026
@author: Usha
"""
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# Load models
diabetes_model = pickle.load(open(
    'C:/Users/Usha/OneDrive/Documents/OneDrive/Documents/multiplepreiction/savedmodels/trained_model.sav','rb'))
breast_cancer_model = pickle.load(open(
    'C:/Users/Usha/OneDrive/Documents/OneDrive/Documents/multiplepreiction/savedmodels/breastcancerclassifier.sav','rb'))
heart_disease_model = pickle.load(open(
    'C:/Users/Usha/OneDrive/Documents/OneDrive/Documents/multiplepreiction/savedmodels/heart_disease.sav','rb'))

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        'Multiple Prediction System',
        ['Diabetes Prediction', 'Breast Cancer Prediction', 'Heart Disease Prediction'],
        default_index=0
    )

# ---------------- Diabetes Page ----------------
if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction")

    pregnancies = st.text_input('Pregnancies')
    glucose = st.text_input('Glucose')
    blood_pressure = st.text_input('Blood Pressure')
    skin_thickness = st.text_input('Skin Thickness')
    insulin = st.text_input('Insulin')
    bmi = st.text_input('BMI')
    dpf = st.text_input('Diabetes Pedigree Function')
    age = st.text_input('Age')

    if st.button("Predict Diabetes"):
        input_data = [int(pregnancies), int(glucose), int(blood_pressure),
                      int(skin_thickness), int(insulin), float(bmi),
                      float(dpf), int(age)]
        input_array = np.asarray(input_data).reshape(1,-1)
        prediction = diabetes_model.predict(input_array)
        st.success("Diabetic" if prediction[0] == 1 else "Not Diabetic")

# ---------------- Breast Cancer Page ----------------
elif selected == 'Breast Cancer Prediction':
    st.title("Breast Cancer Prediction")

    # Example: only a few features for demo, you can add all 30
    mean_radius = st.text_input('Mean Radius')
    mean_texture = st.text_input('Mean Texture')
    mean_perimeter = st.text_input('Mean Perimeter')
    mean_area = st.text_input('Mean Area')
    mean_smoothness = st.text_input('Mean Smoothness')

    if st.button("Predict Breast Cancer"):
        # For demo, fill missing features with zeros
        input_data = [float(mean_radius), float(mean_texture),
                      float(mean_perimeter), float(mean_area),
                      float(mean_smoothness)] + [0]*25
        input_array = np.asarray(input_data).reshape(1,-1)
        prediction = breast_cancer_model.predict(input_array)
        st.success("Malignant" if prediction[0] == 0 else "Benign")

# ---------------- Heart Disease Page ----------------
elif selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction")

    age = st.text_input('Age')
    sex = st.text_input('Sex (1=male,0=female)')
    cp = st.text_input('Chest Pain Type')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Cholesterol')
    fbs = st.text_input('Fasting Blood Sugar')
    restecg = st.text_input('Rest ECG')
    thalach = st.text_input('Max Heart Rate')
    exang = st.text_input('Exercise Induced Angina')
    oldpeak = st.text_input('Oldpeak')
    slope = st.text_input('Slope')
    ca = st.text_input('CA')
    thal = st.text_input('Thal')

    if st.button("Predict Heart Disease"):
        input_data = [int(age), int(sex), int(cp), int(trestbps), int(chol),
                      int(fbs), int(restecg), int(thalach), int(exang),
                      float(oldpeak), int(slope), int(ca), int(thal)]
        input_array = np.asarray(input_data).reshape(1,-1)
        prediction = heart_disease_model.predict(input_array)
        st.success("Heart Disease" if prediction[0] == 1 else "No Heart Disease")
