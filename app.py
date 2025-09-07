# app.py
import streamlit as st
import pandas as pd
import joblib
from agent import agent

#------------------------------
#environment variables
#------------------------------
import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Load model + preprocessor
# -----------------------------
preprocessor = joblib.load("preprocessor.pkl")
clf = joblib.load("baseline_model.pkl")
final_features = joblib.load("final_features.pkl")  # load from training

st.title("🦴 Osteoporosis Risk Prediction & Chat Assistant")

# -----------------------------
# Patient Form
# -----------------------------
with st.form("risk_form"):
    st.subheader("Fill Patient Information")

    age = st.number_input("Age", min_value=18, max_value=100, value=50)
    calcium_in = st.selectbox("Calcium Intake", ["Low", "Adequate", "High"])
    fhistory = st.selectbox("Family History of Osteoporosis", ["No", "Yes"])
    fractures = st.selectbox("Previous Fractures", ["No", "Yes"])
    weight = st.selectbox("Weight", ["Underweight", "Normal", "Overweight"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    hormone = st.selectbox("Hormone", ["Normal", "Postmenopausal"])
    race = st.selectbox("Race", ["Asian", "Caucasian", "African American"])
    activity = st.selectbox("Activity Level", ["Sedentary", "Moderate", "Active"])
    smoking = st.selectbox("Smoking Status", ["No", "Yes"])
    med_condition = st.selectbox("Medical Condition", ["None", "Rheumatoid Arthritis", "Hyperthyroidism"])
    medications = st.selectbox("Medications", ["None", "Corticosteroids"])

    submit = st.form_submit_button("Predict")

# -----------------------------
# Prediction
# -----------------------------
if submit:
    # Build patient dict
    patient_dict = {
        "Age": age,
        "Weight": weight,
        "CalciumIn": calcium_in,
        "Hormone": hormone,
        "FHistory": fhistory,
        "Fractures": fractures,
        "Gender": gender,
        "Race": race,
        "Activity": activity,
        "Smoking": smoking,
        "MedCondition": med_condition,
        "Medications": medications
    }

    # Convert to DataFrame
    patient_df = pd.DataFrame([patient_dict])

    # Transform with preprocessor
    processed = pd.DataFrame(preprocessor.transform(patient_df), columns=final_features)

    # Predict
    prob = clf.predict_proba(processed)[:, 1][0]
    pred = "🟥 High Risk" if prob > 0.7 else "🟩 Low Risk"

    st.subheader("Prediction Result")
    st.write(f"**{pred}** (Probability: {prob:.2f})")

    # Store patient profile in session for chatbot use
    st.session_state["patient"] = patient_dict

# -----------------------------
# Chat Interface
# -----------------------------
st.subheader("💬 Chat with Osteoporosis Agent for Personalized Support")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display past chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me about your osteoporosis risk, prevention, or lifestyle advice..."):
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Attach patient info if available
    patient_info = st.session_state.get("patient", {})
    context = f"Patient info: {patient_info}. User question: {prompt}"

    # Run agent
    response = agent.run(context)

    # Save assistant response
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
