import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset
dataset = load_iris()
X, y = dataset.data, dataset.target

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit App
st.title("Iris Flower Prediction")
st.write("Adjust the sliders to input sepal and petal measurements.")

# User input sliders
sepal_length = st.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Prediction
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_features)[0]
predicted_species = dataset.target_names[prediction]

st.subheader("Prediction")
st.write(f"The predicted iris species is: **{predicted_species}**")

