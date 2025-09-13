# 🦴 OsteoCureAI

**OsteoCureAI** is an innovative Streamlit web application designed to provide insights into bone health and osteoporosis risk. This project integrates two powerful AI-driven features:

1.  **Risk Prediction Model**: Utilizes a machine learning model to predict the risk of osteoporosis based on user-provided health and lifestyle data, offering a personalized risk assessment.
2.  **AI Medical Image Analysis ("OsteoCure Vision")**: An advanced AI assistant that analyzes bone-related medical images (like X-rays and MRIs) to identify potential indicators of bone health issues. It provides patient-friendly explanations and evidence-based information, bridging the gap between clinical reports and patient understanding.

> **Note**: This tool is for informational purposes only and is not a substitute for professional medical advice.

## ✨ Features

- **Osteoporosis Risk Prediction**:
  - Predicts high or low risk of osteoporosis based on health and lifestyle data.
  - Displays the probability of high risk for a personalized assessment.
- **AI-Powered Medical Image Analysis**:
  - Upload bone-related medical images (X-rays, MRIs) for analysis by the "OsteoCure Vision" agent.
  - Receive structured reports with key findings and cautious diagnostic assessments.
  - Get patient-friendly explanations that translate complex medical jargon.
  - Access evidence-based information and research context for findings.

## ⚙️ Setup and Installation

To run this application locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/cosmicmuse666/osteocure
    cd osteocure
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1

    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Running the Application

Once the setup is complete, you can run the Streamlit app with the following command:

```bash
streamlit run app.py
```
