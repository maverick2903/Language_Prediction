import pickle
import streamlit as st
import numpy as np

with open('./model.pkl','rb') as f:
    model = pickle.load(f)

with open('./vectorizer.pkl','rb') as f:
    vectorizer = pickle.load(f)

def predict(input):
    inputArray = np.array([input])
    vectorizedInput = vectorizer.transform(inputArray)
    prediction = model.predict(vectorizedInput)
    return prediction


st.title("Language Prediction App")
input = st.text_input("Enter a sentence in any language:")
if st.button("Predict Language"):
    prediction = predict(input)
    pred = "".join(prediction)
    st.write('Prediction: ',pred)