import streamlit as st
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the trained model
try:
    model = joblib.load('titanic_survival_model.pkl')
except FileNotFoundError:
    st.error("Model file 'titanic_survival_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

# Streamlit App Title
st.title('Titanic Survival Prediction')

st.write("Enter passenger details to predict survival.")

# Input fields for passenger features
pclass_options = {1: '1st Class', 2: '2nd Class', 3: '3rd Class'}
pclass_selection = st.selectbox('Passenger Class (Pclass)', options=list(pclass_options.keys()), format_func=lambda x: pclass_options[x])
sex_options = {'male': 'Male', 'female': 'Female'}
sex_selection = st.selectbox('Sex', options=list(sex_options.keys()), format_func=lambda x: sex_options[x])
age = st.slider('Age', 0, 80, 25)
sibspouse = st.slider('Number of Siblings/Spouses Aboard (sibspouse)', 0, 8, 0)
parentchild = st.slider('Number of Parents/Children Aboard (parentchild)', 0, 6, 0)
fare = st.number_input('Fare', min_value=0.0, value=30.0)

# Create a DataFrame from inputs
input_data = pd.DataFrame({
    'Pclass': [pclass_selection],
    'Sex': [sex_selection],
    'Age': [age],
    'sibspouse': [sibspouse],
    'parentchild': [parentchild],
    'Fare': [fare]
})

# Prediction
if st.button('Predict Survival'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.success(f'The passenger is predicted to **survive**! (Probability: {prediction_proba[0][1]*100:.2f}%)')
    else:
        st.error(f'The passenger is predicted to **not survive**. (Probability: {prediction_proba[0][0]*100:.2f}%)')

    st.write('---')
    st.write('Input Data:')
    st.dataframe(input_data)
