import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.markdown("""
<style>
    body {
      background-color: #000000;
    }
</style>
    """, unsafe_allow_html=True)

st.title('Prediction Salary of Programmers in site StackOverflow')

Country = ['United Kingdom of Great Britain and Northern Ireland', 'Israel',
           'Netherlands', 'United States of America', 'Czech Republic', 'Austria',
           'Italy', 'Canada', 'Germany', 'Poland',
           'Norway', 'France', 'Brazil', 'Sweden',
           'Spain', 'Turkey', 'India', 'Belgium', 'Mexico', 'Switzerland', 'South Africa',
           'Finland', 'Denmark', 'Australia', 'Greece', 'Portugal',
           'Iran, Islamic Republic of...', 'Russian Federation', 'Pakistan',
           'New Zealand']

EdLevel = ['Master’s degree', 'Bachelor’s degree', 'Less than a Bachelors', 'Post grad']

country = st.selectbox(
    'Select Your Country',
    set(Country))

st.write('You selected:', country)

edlevel = st.selectbox(
    'Select Your EdLevel',
    set(EdLevel))

st.write('You selected:', edlevel)

YearsCodePro = st.slider(
    "Select a range of value YearsCodePro",
    0.5, 50.0)
st.write("Values:", YearsCodePro)


def RunFunc(arr):
    model = joblib.load('gbr_model_stackoverflow.pkl')
    d = pd.DataFrame([arr], columns=['Country', 'EdLevel', 'YearsCodePro'])
    price_predict = model.predict(d)
    return price_predict


arr = np.array([country, edlevel, YearsCodePro])
if st.button('Submit'):
    rp = RunFunc(arr)
    result = 'System Predict Sales of Programmer in StackOverflow: {:,.1f},'.format(round(rp[0], 0))[:-3]
    st.success(result)
