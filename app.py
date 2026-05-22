import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------
# Create Sample BP Dataset
# -----------------------------------
data = {
    "BP": [70, 75, 80, 85, 90, 95, 100, 105, 110, 115,
           120, 125, 130, 135, 140, 145, 150, 155, 160],

    "Category": [
        "Low", "Low", "Low",
        "Normal", "Normal", "Normal", "Normal",
        "Normal", "Normal", "Normal",
        "Normal",
        "High", "High", "High",
        "High", "High", "High",
        "High", "High"
    ]
}

df = pd.DataFrame(data)

# -----------------------------------
# Convert Labels
# -----------------------------------
category_map = {
    "Low": 0,
    "Normal": 1,
    "High": 2
}

reverse_map = {
    0: "Low",
    1: "Normal",
    2: "High"
}

df["Target"] = df["Category"].map(category_map)

# -----------------------------------
# Features and Target
# -----------------------------------
X = df[["BP"]]
y = df["Target"]

# -----------------------------------
# Split Dataset
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# Train Logistic Regression
# -----------------------------------
model = LogisticRegression()

model.fit(X_train, y_train)

# -----------------------------------
# Accuracy
# -----------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.title("Blood Pressure Prediction App")

st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# -----------------------------------
# User Details
# -----------------------------------
name = st.text_input("Enter Your Name")

age = st.number_input(
    "Enter Your Age",
    min_value=1,
    max_value=100,
    step=1
)

gender = st.selectbox(
    "Select Gender",
    ["Male", "Female"]
)

# -----------------------------------
# BP Input
# -----------------------------------
bp_value = st.number_input(
    "Enter Blood Pressure",
    min_value=50,
    max_value=200,
    step=1
)

# -----------------------------------
# Predict
# -----------------------------------
if st.button("Predict"):

    prediction = model.predict([[bp_value]])

    result = reverse_map[prediction[0]]

    st.subheader("Prediction Result")

    st.write("Name:", name)
    st.write("Age:", age)
    st.write("Gender:", gender)
    st.write("Blood Pressure:", bp_value)

    if result == "Low":
        st.warning("Low Blood Pressure")

    elif result == "Normal":
        st.success("Blood Pressure is Normal")

    else:
        st.error("High Blood Pressure")