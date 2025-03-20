import streamlit as st
import pandas as pd
import joblib
import numpy as np


st.title('VEHICLE INSURANCE PREDICTION')

# Read the dataset to fill the values in the drop-down list
df = pd.read_csv('train.csv')

# Create the input fields
Gender = st.selectbox("Gender", pd.unique(df["Gender"]))
Age = st.number_input("Age")
Driving_License = st.number_input("Driving_License")
Region_Code = st.number_input("Region_Code")
Previously_Insured = st.number_input("Previously_Insured")
Vehicle_Age = st.selectbox("Vehicle_Age", pd.unique(df["Vehicle_Age"]))
Vehicle_Damage = st.selectbox("Vehicle_Damage", pd.unique(df["Vehicle_Damage"]))
Annual_Premium = st.number_input("Annual_Premium")
Policy_Sales_Channel = st.number_input("Policy_Sales_Channel")
Vintage = st.number_input("Vintage")

# Convert the input values into a dictionary
inputs = {
    "Gender": Gender,
    "Age": Age,
    "Driving_License": Driving_License,
    "Region_Code": Region_Code,
    "Previously_Insured": Previously_Insured,
    "Vehicle_Age": Vehicle_Age,
    "Vehicle_Damage": Vehicle_Damage,
    "Annual_Premium": Annual_Premium,
    "Policy_Sales_Channel": Policy_Sales_Channel,
    "Vintage": Vintage,
}

# Click Predict button
if st.button("Predict"):
    
        # Load the trained model
        model = joblib.load('hack_1.pkl')
        
        # Convert inputs into a DataFrame
        X_input = pd.DataFrame(inputs, index=[0])
        
        # Replace value like Trained data
        X_input['Gender'].replace({ 'Female' : 0 , 'Male' : 1 }, inplace=True )
        X_input['Vehicle_Age'].replace({ '< 1 Year': 0, '1-2 Year' : 1, '> 2 Years': 2 }, inplace=True)
        X_input['Vehicle_Damage'].replace({'No': 0 , 'Yes': 1}, inplace=True)
        
        # Debugging: Display processed input data
        st.write("Processed Input Data:")
        st.write(X_input)
        
        # Make prediction
        prediction = model.predict(X_input)
        
        # Display the prediction
        st.write(f"Prediction: {prediction[0]}")
    
    