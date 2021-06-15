import pandas as pd
import streamlit as st
import os
import sys

penguin=pd.read_csv("penguins.csv")
df = penguin.copy()

target = 'species'
encode = ['sex','island']

for value in encode:
    dummy = pd.get_dummies(df[value], prefix=value)
    df = pd.concat([df,dummy], axis=1)
    del df[value]

class_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}

def target_encode(val):
    return class_mapper[val]

df['species'] = df['species'].apply(target_encode) 
    

#Separate X and Y
X = df.drop('species', axis=1)
Y = df.iloc[:, 0]

#Build RF classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

#Save the classifier
import pickle
pickle.dump(clf, open('penguin_clf.pkl','wb'))
