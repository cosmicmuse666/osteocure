# app.py
import streamlit as st
import pandas as pd
import joblib
from medical_image_analysis_agent import analyze_medical_image
from prediction_agent import agent_chat

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

st.markdown("<h1 style='text-align: center;'>🦴 OsteoCureAI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Osteoporosis risk Prediction and Medical image processing</h4>", unsafe_allow_html=True)

# Custom CSS for the predict button
st.markdown("""
<style>
    
    button[kind="form_submit"] {
        background-color: #FF6347; 
        color: white;
        border: 1px solid #FF6347;
    }
</style>
""", unsafe_allow_html=True)

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

# Create a placeholder for the prediction result
prediction_placeholder = st.empty()

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
    prediction = clf.predict(processed)[0]

    # Display result in the placeholder
    with prediction_placeholder.container():
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("🟥 **High Risk : 1**")
        else:
            st.success("🟩 **Low Risk : 0**")

    # Store patient profile in session for chatbot use
    st.session_state["patient"] = patient_dict

# -----------------------------
# Chat Interface
# -----------------------------
st.subheader("💬 Chat with Osteocure Agent")

# Image upload feature
with st.sidebar:
    st.header("Image Analysis")
    uploaded_file = st.file_uploader(
        "Upload a bone X-ray or MRI to analyze",
        type=["png", "jpg", "jpeg"],
        key="file_uploader"
    )
    analyze_button = st.button("Analyze Image", use_container_width=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Handle image analysis when the button is clicked
if analyze_button and uploaded_file is not None and uploaded_file.file_id != st.session_state.get("last_file_id"):
    st.session_state["last_file_id"] = uploaded_file.file_id
    # Save the uploaded file to a temporary path to be displayed and analyzed
    temp_image_path = os.path.join(".", uploaded_file.name)
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Add a user message to the chat that includes the image
    st.session_state.messages.append({"role": "user", "content": f"Please analyze this uploaded image:", "image": temp_image_path})
    
    # Run the analysis and add its result to the chat history
    with st.spinner("OsteoCure Vision is analyzing the image..."):
        analysis_result = analyze_medical_image(temp_image_path)
        st.session_state.messages.append({"role": "assistant", "content": analysis_result})
        # The temp file is removed here after analysis, but the path is stored in session_state
        # The display loop will handle the cleanup after rendering.
    st.rerun() # Rerun to display the new messages immediately

# Display past chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If the message has an image, display it and then clean up.
        if "image" in msg and os.path.exists(msg["image"]):
            st.image(msg["image"])
            os.remove(msg["image"])

# Chat input
if prompt := st.chat_input("Ask me about your osteoporosis risk, prevention, or lifestyle advice..."):
    # Add user message to chat history and display it
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("CureAI is thinking..."):
            # Attach patient info if available
            patient_info = st.session_state.get("patient", {})

            # Use memory-enabled agent response
            response = agent_chat(prompt, patient_info)

            # Display assistant response
            st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
