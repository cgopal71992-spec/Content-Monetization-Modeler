import streamlit as st
import pandas as pd
import joblib

# Load the trained Ridge Regression model
model = joblib.load("ridge_revenue_model.pkl")

# Page configuration
st.set_page_config(
    page_title="YouTube Ad Revenue Predictor",
    page_icon="▶",
    layout="centered"
)

st.title("YouTube Ad Revenue Predictor")
st.write("Enter video details to estimate expected YouTube ad revenue.")

# User input section
st.subheader("Video Performance Details")

views = st.number_input("Views", min_value=1, value=10000, step=100)
likes = st.number_input("Likes", min_value=0, value=1000, step=10)
comments = st.number_input("Comments", min_value=0, value=250, step=10)
watch_time_minutes = st.number_input(
    "Total Watch Time (Minutes)",
    min_value=0.0,
    value=37522.22,
    step=100.0
)
video_length_minutes = st.number_input(
    "Video Length (Minutes)",
    min_value=0.1,
    value=10.0,
    step=0.5
)
subscribers = st.number_input(
    "Channel Subscribers",
    min_value=0,
    value=500000,
    step=1000
)

# Categorical inputs
category = st.selectbox(
    "Video Category",
    ["Education", "Entertainment", "Gaming", "Lifestyle", "Music", "Tech"]
)
device = st.selectbox(
    "Device",
    ["Desktop", "Mobile", "Tablet", "TV"]
)
country = st.selectbox(
    "Country",
    ["CA", "DE", "IN", "UK", "US"]
)
day_of_week = st.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
month = st.selectbox("Month", list(range(1, 13)), index=0)

# Calculate the same feature used during model training
engagement_rate = (likes + comments) / views

# Make prediction when button is clicked
if st.button("Predict Ad Revenue"):
    input_data = pd.DataFrame({
        "views": [views],
        "likes": [likes],
        "comments": [comments],
        "watch_time_minutes": [watch_time_minutes],
        "video_length_minutes": [video_length_minutes],
        "subscribers": [subscribers],
        "category": [category],
        "device": [device],
        "country": [country],
        "month": [month],
        "day_of_week": [day_of_week],
        "engagement_rate": [engagement_rate]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated YouTube Ad Revenue: ${prediction:,.2f}")

    st.caption(
        "This is a machine-learning estimate based on the provided video metrics."
    )

    #I developed a YouTube ad revenue prediction system using five regression models. After data cleaning, feature engineering, and EDA, Ridge Regression was selected as the final model because it achieved the best RMSE of 13.48, MAE of 3.12, and R² score of 0.9526. Watch time was identified as the strongest revenue predictor. A Streamlit application was built to provide interactive revenue predictions.