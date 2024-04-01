#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:03:01 2024

@author: NonsoIkenwa
"""

import numpy as np
import pickle
import streamlit as st

#Loading the model
loaded_model = pickle.load(open('/Users/NonsoIkenwa/Desktop/ML projects/Diabetes prediction/trained_model.sav','rb'))

#Creating a funtion for predition 
def Diabetes_prediction(input_data):
    
    #Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    #standardize the input data
    #std_data = scaler.transform(input_data_reshaped)
    #print(std_data)
   


    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0]==0):
        return'Not diabetic'
    else:
        return'The person is diabetic'
    
    
    
def main():
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    
    #Getting input data from the user
    #Create all the variables we need
    #Pregnancies,Glucose,BloodPressure,SkinThickness	,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    # Unique key for each text_input widget
    Pregnancies_key = "pregnancies_input"
    Pregnancies = st.text_input('Number of Pregnancies', key=Pregnancies_key)
    
    # Other widgets with unique keys
    Glucose_key = "Glucose_input"
    Glucose = st.text_input('Glucose Level', key=Glucose_key)
    
    BloodPressure_key = "Bloodpressure_input"
    BloodPressure = st.text_input('Blood Pressure', key=BloodPressure_key)
    
    SkinThickness_key = "SkinThickness_input"
    SkinThickness = st.text_input('Skin Thickness', key=SkinThickness_key)
    
    Insulin_key = "Insulin level"
    Insulin = st.text_input('Insulin Level', key=Insulin_key)
    
    BMI_key = "BMI_input"
    BMI = st.text_input('BMI', key=BMI_key)
    
    DiabetesPedigreeFunction_key = "DiabetesPedigreeFunction_input"
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', key=DiabetesPedigreeFunction_key)
    
    Age_key = "Age_input"
    Age = st.text_input('Age Level', key=Age_key)
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = Diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    
    st.success(diagnosis)
    
    
    
    
if __name__ == '__main__':
    main()