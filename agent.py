import pandas as pd
import joblib
from langchain import OpenAI, LLMMathChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
import getpass
import os



# Load LLM 
load_dotenv()
# Ensure GOOGLE_API_KEY is set
# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Ensure OPENAI_API_KEY is set
# if "OPENAI_API_KEY" not in os.environ:
#     raise ValueError("⚠️ Missing OPENAI_API_KEY. Please set it in your .env or Streamlit secrets.")

# Initialize LLM

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0,
    max_tokens=4096,
    timeout=None,
    max_retries=2,
   
)

# -------------------------
# Tool 1: Prediction Tool
# -------------------------
def predict_osteoporosis(features):
    """Takes patient features (dict), runs model, returns risk level."""
    model = joblib.load("baseline_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    
    df = pd.DataFrame([features])
    processed = preprocessor.transform(df)
    prediction = model.predict(processed)[0]
    prob = model.predict_proba(processed)[:, 1][0]

    risk = "High Risk" if prediction > 0.7 else "Low Risk"
    return f"{risk} (Probability: {prob:.2f})"

prediction_tool = Tool(
    name="Osteoporosis Risk Predictor",
    func=lambda x: predict_osteoporosis(eval(x)),
    description="Predicts osteoporosis risk (High/Low) from patient health inputs"
)

# -------------------------
# Tool 2: Math / reasoning
# -------------------------
# math_tool = LLMMathChain.from_llm(llm=llm)

# -------------------------
# Instruction for the Agent
# -------------------------
AGENT_INSTRUCTION = """
You are a medical assistant specializing in osteoporosis.
When given patient details:
1. Use the risk predictor tool to determine osteoporosis risk.
2. Explain the prediction in clear, human-friendly language.
3. Suggest *preventive measures* tailored to the patient’s risk:
   - If High Risk: recommend calcium/vitamin D intake, resistance training, lifestyle changes, medical screening.
   - If Low Risk: recommend healthy diet, regular weight-bearing exercise, avoiding smoking & excess alcohol.
4. Always sound encouraging and supportive, not alarming.
"""

# -------------------------
# Initialize Agent
# -------------------------
agent_kwargs = {"prefix": AGENT_INSTRUCTION}
agent = initialize_agent(
    tools=[prediction_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs=agent_kwargs,
    handle_parsing_errors=True, # Handle cases where the LLM output is not a valid action
)
