import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

st.title("Breast Cancer Detection")
st.write("Enter tumor measurements to predict malignancy")

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

st.sidebar.header("Tumor Measurements")
radius = st.sidebar.slider("Mean Radius", 6.0, 30.0, 14.0)
texture = st.sidebar.slider("Mean Texture", 9.0, 40.0, 19.0)
perimeter = st.sidebar.slider("Mean Perimeter", 40.0, 190.0, 90.0)

mean_values = X_train.mean(axis=0)
input_data = mean_values.copy()
input_data[0] = radius
input_data[1] = texture
input_data[2] = perimeter

if st.sidebar.button("Predict"):
    prediction = model.predict([input_data])
    prob = model.predict_proba([input_data])
    
    if prediction[0] == 0:
        st.error(f"Malignant (Cancerous) - Probability: {prob[0][0]:.2%}")
    else:
        st.success(f"Benign (Non-cancerous) - Probability: {prob[0][1]:.2%}")

st.write(f"Model Accuracy: {(model.score(X_test, y_test)*100):.1f}%")