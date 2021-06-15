import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pickle

st.write("""
# Penguin Prediction App

This app is used to predict the species of penguin!

The dataset is obtained from Kaggle: https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data

App created by Viraj Kulkarni.
         """)
         
st.sidebar.header("User Input Features")

#need to add link later of example
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/viraj-kulkarni952/Penguin-Prediction-App/main/penguins_example.csv)
""")

#Collects user input features into dataframe
#If the dataframe is NOT empty, then app will use the user input dataframe, otherwise it uses manual input features
user_input_csv = st.sidebar.file_uploader("Upload your CSV file here", type=['csv'])
if user_input_csv is not None:
    input_dataframe = pd.read_csv(user_input_csv)
else:
    def user_input_features():
        island = st.sidebar.selectbox("Island", ['Biscoe','Dream','Torgersen'])
        sex = st.sidebar.selectbox("Sex",['MALE','FEMALE'])
        bill_length = st.sidebar.slider("Bill length (mm)", 32.1, 59.6, 45.7)
        bill_depth = st.sidebar.slider("Bill depth (mm)", 13.1, 21.5, 17.3)
        flipper_length = st.sidebar.slider("Flipper length (mm)", 172, 231, 201)
        body_mass = st.sidebar.slider("Body mass (g)", 2700, 6300, 4500)
        data = {'island': [island],
                'culmen_length_mm': [bill_length],
                'culmen_depth_mm': [bill_depth],
                'flipper_length_mm': [flipper_length],
                'body_mass_g': [body_mass],
                'sex': [sex]}
        features = pd.DataFrame(data)
        return features
    input_dataframe=user_input_features()
    
    
#Combine user input with the penguin dataset
penguin_raw = pd.read_csv('penguins.csv')
penguins = penguin_raw.drop(columns=['species'])
df = pd.concat([input_dataframe,penguins],axis=0)

#Encoding the categorical features
encode = ['sex','island']

for value in encode:
    dummy = pd.get_dummies(df[value], prefix=value)
    df = pd.concat([df,dummy], axis=1)
    del df[value]
    
df = df[:1]

#Display User Input
st.subheader("User Input Features")

if user_input_csv is not None:
    st.write(df)
else:
    st.write("Awaiting for user input of CSV file. Currently using the example input parameters.")
    st.write(df)
        
#Read in classification model from model-builder
load_clf = pickle.load(open('penguin_clf.pkl', 'rb'))

#Apply the model to make predictions
pred = load_clf.predict(df)
pred_probability = load_clf.predict_proba(df)

#Change column names for pred probability
pred_probability = pd.DataFrame(pred_probability)
column_name_predprob = ["Adelie","Chinstrap","Gentoo"]
pred_probability.columns = column_name_predprob

#Put pred on using streamlit
st.subheader("Prediction")
penguin_prediction = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguin_prediction[pred])

#Put pred_probability on using streamlit
st.subheader("Prediction Probability")
st.write(pred_probability)
st.write("")






