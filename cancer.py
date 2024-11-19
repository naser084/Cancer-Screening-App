import streamlit as st
import joblib
import tensorflow as tf
import pandas as pd
import time

# Load the scaler and the Keras model
try:
    scaler = joblib.load('scaler.h5')  # Ensure this is the correct path to your scaler file
except FileNotFoundError:
    st.error("Scaler file not found. Please ensure 'scaler.pkl' is available.")

try:
    model = tf.keras.models.load_model('cancer_prediction_model.h5')  # Ensure this is the correct path to your model file
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'new_cancer_prediction_model.h5' is available.")

# CSS styles to enhance the look of the app with specific colors
st.markdown(
    """
    <style>
    /* Global styles */
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f8f9fa;  /* Very light gray background for the page */
    }

    /* Main container styling */
    .block-container {
        max-width: 800px; /* Limit max width of the main container */
        padding: 20px;
        background-color: #8F00FF; /* White background for main content */
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); /* Soft shadow for main container */
        margin: auto;
    }

    /* Title styling */
    .css-18e3th9 h1 {
        color: #8F00FF;  /* Light green color for the main title */
        font-size: 2.8em;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 20px;
    }

    /* Header styling */
    .css-18e3th9 h2, .css-18e3th9 h3 {
        color: #8F00FF;  /* Light green for headers */
        font-size: 1.8em;
        text-align: center;
        margin-bottom: 15px;
    }

    /* Sidebar styling */
    .css-1aumxhk {
        background-color: #1E90FF; /* Light blue background for sidebar */
        border-radius: 10px;
        padding: 20px;
        color: #1E90FF;  /* Dark blue text color */
        font-size: 1.1em;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); /* Soft shadow for sidebar */
    }

    /* Sidebar headers */
    .css-1aumxhk h3, .css-1aumxhk h2 {
        color: #1E90FF;  /* Medium blue for sidebar headers */
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    /* Style for the sidebar text */
    .css-1aumxhk p {
        color: #40E0D0;  /* Dark blue for sidebar text */
    }

    /* Button styles */
    .css-1lcbg1e {
        background-color: #40E0D0;  /* Button background color - light green */
        color: white;  /* Text color */
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-size: 1.1em;
    }
    .css-1lcbg1e:hover {
        background-color: #40E0D0; /* Hover effect with slightly darker green */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main application
st.title("Cancer Detection by Naser")
st.header("Check if your tumor is cancerous or not.")

# Sidebar with sections
st.sidebar.title("Information")
st.sidebar.subheader("1. How Cancer Occurs")
st.sidebar.write(
    "Cancer occurs when there are abnormal and uncontrolled cell divisions in the body. "
    "These cells can form a mass called a tumor, which may be benign (non-cancerous) or malignant (cancerous). "
    "Malignant tumors can invade nearby tissues or spread to other parts of the body, causing serious health risks."
)

st.sidebar.subheader("2. Tumor Cancer Detector")
st.sidebar.write(
    "The Tumor Cancer Detector uses machine learning algorithms to analyze input features from a patient. "
    "Based on the data provided, it predicts whether the tumor is likely to be cancerous or not."
)

st.sidebar.subheader("3. Problem Statement")
st.sidebar.write(
    "Early detection of cancer is crucial for effective treatment and increased survival rates. "
    "Traditional diagnostic methods can be time-consuming and expensive. "
    "Our goal is to develop a fast, accurate, and cost-effective solution to predict whether a tumor is cancerous."
)

st.sidebar.subheader("4. Solution Overview")
st.sidebar.write(
    "This application uses a machine learning model trained on the Breast Cancer dataset. "
    "It analyzes various tumor features, such as size, texture, and shape, to predict the likelihood of cancer. "
    "The model is built using deep learning techniques for high accuracy and reliability."
)

# Input fields organized into three columns
col1, col2, col3 = st.columns(3)

# Feature inputs using select sliders
mean_radius = col1.select_slider("Mean Radius", options=[round(x, 1) for x in list(range(60, 301, 1))], value=140) / 10
mean_texture = col2.select_slider("Mean Texture", options=[round(x, 1) for x in list(range(90, 401, 1))], value=200) / 10
mean_perimeter = col3.select_slider("Mean Perimeter", options=[round(x, 1) for x in list(range(400, 2001, 1))], value=900) / 10
mean_area = col1.select_slider("Mean Area", options=[x for x in range(100, 3001, 50)], value=600)
mean_smoothness = col2.select_slider("Mean Smoothness", options=[round(x, 3) for x in list(range(50, 201))], value=100) / 1000
mean_compactness = col3.select_slider("Mean Compactness", options=[round(x, 2) for x in list(range(2, 36))], value=10) / 100
mean_concavity = col1.select_slider("Mean Concavity", options=[round(x, 2) for x in list(range(0, 51))], value=8) / 100
mean_concave_points = col2.select_slider("Mean Concave Points", options=[round(x, 2) for x in list(range(0, 21))], value=5) / 100
mean_symmetry = col3.select_slider("Mean Symmetry", options=[round(x, 2) for x in list(range(10, 41))], value=18) / 100
mean_fractal_dimension = col1.select_slider("Mean Fractal Dimension", options=[round(x, 2) for x in list(range(50, 101))], value=60) / 1000
radius_error = col2.select_slider("Radius Error", options=[round(x, 1) for x in list(range(10, 301, 1))], value=30) / 100
texture_error = col3.select_slider("Texture Error", options=[round(x, 1) for x in list(range(50, 501, 1))], value=150) / 100
perimeter_error = col1.select_slider("Perimeter Error", options=[round(x, 1) for x in list(range(5, 101, 1))], value=20) / 10
area_error = col2.select_slider("Area Error", options=[x for x in range(5, 101, 5)], value=15)
smoothness_error = col3.select_slider("Smoothness Error", options=[round(x, 3) for x in list(range(1, 11))], value=5) / 1000
compactness_error = col1.select_slider("Compactness Error", options=[round(x, 3) for x in list(range(5, 51))], value=20) / 1000
concavity_error = col2.select_slider("Concavity Error", options=[round(x, 3) for x in list(range(5, 51))], value=20) / 1000
concave_points_error = col3.select_slider("Concave Points Error", options=[round(x, 3) for x in list(range(1, 21))], value=10) / 1000
symmetry_error = col1.select_slider("Symmetry Error", options=[round(x, 3) for x in list(range(10, 51))], value=30) / 1000
fractal_dimension_error = col2.select_slider("Fractal Dimension Error", options=[round(x, 3) for x in list(range(1, 6))], value=2) / 1000
worst_radius = col3.select_slider("Worst Radius", options=[round(x, 1) for x in list(range(70, 401, 1))], value=160) / 10
worst_texture = col1.select_slider("Worst Texture", options=[round(x, 1) for x in list(range(120, 501, 1))], value=250) / 10
worst_perimeter = col2.select_slider("Worst Perimeter", options=[round(x, 1) for x in list(range(500, 2501, 10))], value=1000) / 10
worst_area = col3.select_slider("Worst Area", options=[x for x in range(200, 4001, 50)], value=700)
worst_smoothness = col1.select_slider("Worst Smoothness", options=[round(x, 2) for x in list(range(5, 31))], value=15) / 100
worst_compactness = col2.select_slider("Worst Compactness", options=[round(x, 2) for x in list(range(2, 51))], value=25) / 100
worst_concavity = col3.select_slider("Worst Concavity", options=[round(x, 2) for x in list(range(0, 101))], value=20) / 100
worst_concave_points = col1.select_slider("Worst Concave Points", options=[round(x, 2) for x in list(range(0, 31))], value=10) / 100
worst_symmetry = col2.select_slider("Worst Symmetry", options=[round(x, 2) for x in list(range(10, 61))], value=25) / 100
worst_fractal_dimension = col3.select_slider("Worst Fractal Dimension", options=[round(x, 2) for x in list(range(50, 151))], value=80) / 1000

# Input list for the model
user_inputs = [
    mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity,
    mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error,
    area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error,
    fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
    worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension
]

# Prediction function
def predict_cancer(inputs):
    try:
        # Convert input list to DataFrame
        input_df = pd.DataFrame([inputs], columns=[
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
            'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
            'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ])

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)
        return "Cancerous" if prediction[0][0] > 0.5 else "Not Cancerous"
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        return None

# Predict button
if st.button("Predict"):
    
    result = predict_cancer(user_inputs)
    if result == "Cancerous":
        st.error("The tumor is Cancerous.Get checked by a doctor right now!")
        st.snow()
    elif result == "Not Cancerous":
        st.success("The tumor is NOT Cancerous.")
        st.balloons()

st.header("Rate my App")
rating = st.select_slider("Rating", ["Worst", "Bad","Average", "Good", "Excellent"])

# Define the color based on the rating
if   rating == "Worst":
     color = "yellow"
elif rating == "Bad":
     color = "orange"
elif rating =="Average":  
     color=  "pink"
elif rating == "Good":
     color = "green"
elif rating == "Excellent":
     color = "blue"

# Display the selected rating with the corresponding color
st.markdown(
    f"<h3 style='color: {color};'>Your rating: {rating}</h3>",
    unsafe_allow_html=True
)

# Submit button for rating
if st.button("Submit Rating"):  # Unique label to avoid conflicts
    st.success("Rating submitted successfully!")
    st.snow()  # Shows snow effect on successful submission
