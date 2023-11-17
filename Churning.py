import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Load the model outside the function to avoid loading it on every prediction
model = joblib.load('Churning customers_Ass.joblib')
model_scaler = joblib.load('scaller.joblib')  # Corrected the scaler file name

st.title('Churn Prediction App')

online_security = st.text_input('Online Security', ('Yes', 'No', 'No internet service'))
device_protection = st.text_input('Device Protection', ('Yes', 'No', 'No internet service'))
tech_support = st.text_input('Tech Support', ('Yes', 'No'))
streaming_tv = st.text_input('Streaming TV', ('Yes', 'No'))
streaming_movies = st.text_input('Streaming Movies', ('Yes', 'No'))
contract = st.text_input('Contract', ('Month-to-month', 'One year', 'Two year'))
payment_method = st.text_input('Payment Method', ('Electronic check', 'Bank transfer', 'Credit card/debit card'))
tenure = st.number_input('Tenure', min_value=0, max_value=100)
monthly_charges = st.number_input('Monthly Charges', min_value=0)
total_charges = st.number_input('Total Charges', min_value=0)

def predict(tenure, monthly_charges, total_charges, online_security, device_protection, streaming_tv, tech_support, contract, payment_method):
    # Normalize the input data
    data = {
        'OnlineSecurity': online_security,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaymentMethod': payment_method,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,    
    }

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([data])

    inputs = np.array(['OnlineSecurity','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod','tenure','MonthlyCharges','TotalCharges'])

    inputs = inputs.reshape(1,-1)
        
    #Scaling the features 

    scaled_features= model_scaler.transform(inputs)

    #Prediction

    prediction = model.predict(scaled_features)[0]

if st.button('Predict Churn Probability'):
    prediction = predict(tenure, monthly_charges, total_charges, online_security, device_protection, streaming_tv, tech_support, contract, payment_method)
    prediction_probability = prediction * 100

    # Save the prediction result to a file
    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(f'Churn Probability: {prediction_probability}%')

    st.success(f'Churn Probability: {prediction_probability}% Result saved to "output.txt"')
