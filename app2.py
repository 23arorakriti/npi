import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("npi_data.csv")
    
    # Convert time columns to datetime
    df["Login Time"] = pd.to_datetime(df["Login Time"])
    df["Logout Time"] = pd.to_datetime(df["Logout Time"])
    
    return df

df = load_data()

# Feature Engineering
def preprocess_data(df):
    df["Login Hour"] = df["Login Time"].dt.hour
    df["Usage Category"] = pd.cut(df["Usage Time (mins)"], bins=[0, 30, 60, 120, np.inf], labels=[1, 2, 3, 4])
    
    # Encode categorical variables
    le_speciality = LabelEncoder()
    le_region = LabelEncoder()
    
    df["Speciality"] = le_speciality.fit_transform(df["Speciality"])
    df["Region"] = le_region.fit_transform(df["Region"])

    return df, le_speciality, le_region

df, le_speciality, le_region = preprocess_data(df)

# Train ML Model (Random Forest)
def train_model(df):
    X = df[["Login Hour", "Usage Category", "Speciality", "Region", "Count of Survey Attempts"]]
    y = (df["Count of Survey Attempts"] > 2).astype(int)  # Label: 1 if more than 2 attempts, else 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

model = train_model(df)

# Save the model for reuse
joblib.dump(model, "doctor_survey_model.pkl")

# Function to predict the best doctors for a given time range
def get_best_doctors(input_time):
    try:
        input_time = datetime.strptime(input_time, "%H:%M").time()
    except ValueError:
        st.error("⚠️ Invalid time format! Please enter time in HH:MM format.")
        return pd.DataFrame()
    
    start_time = (datetime.combine(datetime.today(), input_time) - timedelta(minutes=15)).time()
    end_time = (datetime.combine(datetime.today(), input_time) + timedelta(minutes=15)).time()
    
    # Generate predictions
    df["Predicted_Attendance"] = model.predict(df[["Login Hour", "Usage Category", "Speciality", "Region", "Count of Survey Attempts"]])
    
    # Filter doctors based on time range & model prediction
    filtered_df = df[
    (df["Login Time"].dt.time <= end_time) &  # Logged in before or at end_time
    (df["Logout Time"].dt.time >= start_time) &  # Logged out after or at start_time
    (df["Predicted_Attendance"] == 1)
    ] 
    
    return filtered_df

# Streamlit UI
st.title("📩 AI-Powered NPI Survey Targeting")

# User input for time entry
time_input = st.text_input("Enter a time like (7:55):")

if st.button("Find Best Doctors"):
    if time_input:
        selected_doctors = get_best_doctors(time_input)
        
        if not selected_doctors.empty:
            st.success(f"✅ {len(selected_doctors)} doctors found who are likely to attend the survey between {time_input} ± 15 minutes.")
            st.dataframe(selected_doctors)
            
            # Download CSV
            csv_data = selected_doctors.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv_data, "filtered_doctors.csv", "text/csv")
        
        else:
            st.warning("⚠️ No doctors found at this time. Try a different time.")
    else:
        st.error("⚠️ Please enter a valid time in HH:MM format.")
