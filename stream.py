import streamlit as st
import pandas as pd
import requests
import json
from typing import Literal

# --- Configuration ---
# NOTE: Ensure your FastAPI server is running at this exact URL before running this app.
FASTAPI_ENDPOINT = "http://127.0.0.1:8000/predict"

# --- Feature Mapping and Constants ---
# These must match the exact values/structure of your FastAPI Inputs class
FEATURE_OPTIONS = {
    "Gender": ['Male', 'Female'],
    "Blood": ['B-', 'A+', 'A-', 'O+', 'AB+', 'AB-', 'B+', 'O-'],
    "condition": ['Cancer', 'Obesity', 'Diabetes', 'Asthma', 'Hypertension', 'Arthritis'],
    "Medication": ['Paracetamol', 'Ibuprofen', 'Aspirin', 'Penicillin', 'Lipitor']
}
PREDICTION_MAPPING = {
    "Normal": ("✅ Low Risk", "success"),
    "Inconclusive": ("⚠️ Uncertain Risk", "warning"),
    "Abnormal": ("❌ High Risk", "error")
}

# --- Streamlit UI ---

st.set_page_config(
    page_title="AI Health Predictor",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🩺 AI Health Risk Checker")
st.markdown("Enter patient data below to get a prediction on the **Test Results** using the deployed **FastAPI** model.")

# --- Form for User Input ---
with st.form("prediction_form"):
    st.subheader("Patient Vitals and Details (7 Features)")

    # Layout two columns for better arrangement
    col1, col2 = st.columns(2)

    # Column 1 Inputs
    age = col1.number_input("Age", min_value=1, max_value=120, value=35, step=1)
    gender = col1.selectbox("Gender", options=FEATURE_OPTIONS["Gender"])
    blood = col1.selectbox("Blood Type", options=FEATURE_OPTIONS["Blood"])
    room = col1.number_input("Room Number", min_value=100, max_value=999, value=345)

    # Column 2 Inputs
    condition = col2.selectbox("Medical Condition", options=FEATURE_OPTIONS["condition"])
    medication = col2.selectbox("Medication", options=FEATURE_OPTIONS["Medication"])
    amount = col2.number_input("Billing Amount", min_value=0.01, value=17695.91, format="%.2f")

    # Submission button
    submitted = st.form_submit_button("Get Prediction")

# --- Prediction Logic ---

if submitted:
    # 1. Create the payload matching the FastAPI Inputs Pydantic schema
    payload = {
        "Age": int(age),
        "Gender": gender,
        "Blood": blood, 
        "condition": condition,
        "Amount": float(amount),
        "Room": int(room),
        "Medication": medication
    }

    try:
        # 2. Send request to FastAPI endpoint
        response = requests.post(
            FASTAPI_ENDPOINT,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )

        # 3. Handle response status
        if response.status_code == 200:
            data = response.json()
            # CRITICAL: Use the exact key returned by the FastAPI endpoint: "HEALTH: "
            predicted_label = data.get("HEALTH: ") 

            if predicted_label:
                display_text, display_type = PREDICTION_MAPPING.get(predicted_label, (f"Unknown Label: {predicted_label}", "info"))
                
                st.subheader("Prediction Result:")
                st.markdown(f"**Predicted Result:** `{predicted_label}`")
                
                # Display result using Streamlit's alert system
                if display_type == "success":
                    st.success(display_text)
                elif display_type == "warning":
                    st.warning(display_text)
                elif display_type == "error":
                    st.error(display_text)
                else:
                    st.info(display_text)
            
            else:
                st.error("Prediction successful, but the expected key ('HEALTH: ') was missing in the response.")
                st.json(data)
        
        elif response.status_code == 422:
            st.error("Validation Error (422): Please check your input values against the allowed options.")
            st.json(response.json())
            
        else:
            st.error(f"Error communicating with the model server. Status code: {response.status_code}")
            st.info("Please ensure the FastAPI server is running and accessible.")
            st.json(response.json())

    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not reach the FastAPI service. Ensure it is running at `http://127.0.0.1:8000`.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
