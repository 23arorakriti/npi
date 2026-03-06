# 🧠 AI-Powered NPI Survey Targeting

This project is a **Machine Learning powered Streamlit application** that helps identify healthcare providers who are most likely to respond to surveys based on their platform usage patterns.

The system analyzes provider activity using **NPI (National Provider Identifier) data** and predicts which doctors are most likely to participate in surveys at a given time.

The goal of this project is to help organizations **target surveys efficiently by identifying the best time and doctors to contact**.

---

# 🚀 Live Demo

🔗 Streamlit App: https://npitest.streamlit.app

---

# 📌 Project Overview

Healthcare platforms often struggle with **low survey participation rates** from providers.

This application uses **machine learning to predict which doctors are likely to attend a survey**, based on historical usage data.

The system considers factors such as:

- Login time
- Usage duration
- Medical specialty
- Geographic region
- Previous survey attempts

Users can input a **specific time**, and the system will return doctors who are **likely to attend a survey within ±15 minutes of that time**.

---

# 📊 Dataset Description

The dataset contains **healthcare provider platform activity logs**.

Each provider is identified using their **NPI (National Provider Identifier)**.

An **NPI** is a **10-digit unique identification number assigned to healthcare providers in the United States**.

### Dataset Columns

| Column | Description |
|------|-------------|
| NPI | Unique identifier for healthcare providers |
| State | US state of the provider |
| Login Time | Time the provider logged into the platform |
| Logout Time | Time the provider logged out |
| Usage Time (mins) | Total time spent on the platform |
| Region | Geographic region |
| Speciality | Medical specialization |
| Count of Survey Attempts | Number of survey attempts made |

---

# ⚙️ Machine Learning Model

The system uses a **Random Forest Classifier** to predict survey attendance.

### Feature Engineering

The following features are created:

- **Login Hour** → extracted from login time
- **Usage Category** → categorized usage duration
- **Speciality Encoding**
- **Region Encoding**

### Prediction Logic

The trained model predicts which doctors are **likely to respond**.

---

# 🧠 How the System Works

1. Dataset is loaded and preprocessed
2. Feature engineering is applied
3. A **Random Forest model** is trained
4. The user enters a time (HH:MM)
5. The system finds doctors logged in **±15 minutes around that time**
6. The model predicts doctors likely to attend surveys
7. Results are displayed in the dashboard

---

# ✨ Features

- AI-powered survey targeting
- Machine learning predictions
- Time-based doctor filtering
- Interactive Streamlit interface
- Download results as CSV
- Automated feature engineering

---

# 🛠 Tech Stack

### Programming
- Python

### Machine Learning
- Scikit-learn
- Random Forest Classifier

### Data Processing
- Pandas
- NumPy

### Model Storage
- Joblib

### Web Interface
- Streamlit

# 👩‍💻 Author

**Kriti Arora**
