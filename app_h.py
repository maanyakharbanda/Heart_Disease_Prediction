import streamlit as st
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('Heart_Disease_Prediction\pipe_rf.pkl', 'rb'))

# Title of the app
st.title('Heart Disease Prediction')

# Input form
st.header("Enter Patient's Health Information")

# Columns for inputs
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age', min_value=1, max_value=120, value=50, step=1)
    trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)

with col2:
    sex = st.selectbox('Sex', ['Male', 'Female'])
    chol = st.number_input('Cholesterol Level (mg/dl)', min_value=100, max_value=600, value=200)
    oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=6.0, value=1.0, step=0.1)

with col3:
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    restecg = st.selectbox('Resting ECG', ['Normal', 'Having ST-T Wave Abnormality', 'Showing Left Ventricular Hypertrophy'])
    slope = st.selectbox('ST Slope', ['Upsloping', 'Flat', 'Downsloping'])
    thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# Additional inputs
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3, 4])  # This is the missing feature

# Convert categorical values to numeric values as expected by the model
sex = 1 if sex == 'Male' else 0
cp = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
restecg = ['Normal', 'Having ST-T Wave Abnormality', 'Showing Left Ventricular Hypertrophy'].index(restecg)
slope = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
thal = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)
fbs = 1 if fbs == 'Yes' else 0
exang = 1 if exang == 'Yes' else 0

# Create input DataFrame
input_data = pd.DataFrame({
    'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
    'chol': [chol], 'fbs': [fbs], 'restecg': [restecg],
    'thalach': [thalach], 'exang': [exang],
    'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
})

# Display input data for confirmation
st.subheader("Input Data Summary")
st.table(input_data)

# Predict and display the result when the button is clicked
if st.button('Predict Heart Disease Risk'):
    prediction = model.predict_proba(input_data)
    no_disease = prediction[0][0]  # Probability of no heart disease
    disease = prediction[0][1]  # Probability of heart disease

    st.header("Prediction Results")
    st.write(f"Probability of Heart Disease: {round(disease * 100, 2)}%")
    st.write(f"Probability of No Heart Disease: {round(no_disease * 100, 2)}%")

    if disease > 0.5:
        st.error("High Risk of Heart Disease Detected.")
    else:
        st.success("Low Risk of Heart Disease Detected.")
