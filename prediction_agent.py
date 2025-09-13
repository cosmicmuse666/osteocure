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
# Tool 3: Tavily Search Tool (using official tavily package)
# -------------------------
# from tavily import TavilyClient
# from langchain.agents import Tool

# import os
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# def tavily_search(query):
#     """Search the web using Tavily and return a summary."""
#     result = tavily_client.search(query)
#     # Tavily returns a dict with 'answer' and 'sources'
#     return result.get("answer", "")

# tavily_search_tool = Tool(
#     name="Tavily Web Search",
#     func=tavily_search,
#     description="Searches the web for up-to-date, reliable information and summarizes the findings."
# )

# -------------------------
# Tool 4: DuckDuckGo Search Tool
# -------------------------
from duckduckgo_search import DDGS
from langchain.agents import Tool

def ddg_search(query):
    """Search the web using DuckDuckGo and return a summary of the top result."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=1))
        if results:
            return results[0].get("body", results[0].get("title", "No summary found."))
        else:
            return "No results found."

ddg_search_tool = Tool(
    name="DuckDuckGo Web Search",
    func=ddg_search,
    description="Searches the web using DuckDuckGo for up-to-date information and summarizes the top result."
)

# -------------------------
# Instruction for the Agent
# -------------------------
AGENT_INSTRUCTION = """
### Role and Goal
You are "CureAI", an expert medical assistant specializing in osteoporosis. Your goal is to provide risk assessment, patient education, and personalized support in an empathetic and clear manner.

### Available Tools
1.  **Osteoporosis Risk Predictor**: Use this tool when you have a patient's health information to calculate their specific risk level.
2.  **DuckDuckGo Web Search**: Use this tool to find answers to general questions about osteoporosis, treatments, lifestyle factors, or any other topic where you need up-to-date, external information.

### Core Workflow
1.  **Analyze the Request**: Understand if the user is asking for a risk prediction, general information, or follow-up advice.
2.  **Use Tools if Necessary**:
    *   If patient data is provided for a prediction, use the **Osteoporosis Risk Predictor** tool first.
    *   If the user asks a question you cannot answer from your internal knowledge, use the **DuckDuckGo Web Search** tool to find a reliable answer.
3.  **Synthesize and Respond**:
    *   **For Risk Predictions**: Clearly state the risk level and probability. Explain what this means in simple terms. Provide personalized, evidence-based recommendations based on their profile (age, gender, lifestyle, etc.).
    *   **For General Questions**: Provide a clear, concise answer based on your knowledge or the search results.
    *   **For All Responses**: Maintain a supportive, empathetic, and non-alarming tone. Encourage positive action and empower the user.

### Important Guardrails
*   **provide a medical diagnosis.** Always frame your output as a risk assessment and educational information.
*   Always relate your advice back to the patient's specific profile if it has been provided.

### Response Format
You MUST strictly follow this format for your reasoning process:
Thought: Your step-by-step reasoning on how to answer the user's request.
Action: The tool to use. This should be one of the tools listed under "Available Tools".
Action Input: The input string for the action.
Observation: The result returned from the tool.
Final Answer: Your complete, well-formatted, and empathetic response to the user.
"""

# -------------------------
# Memory for Chat History
# -------------------------
class ChatMemory:
    def __init__(self):
        self.history = []

    def add(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_context(self):
        # Format history for context string
        return "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.history]
        )

# Create a global memory instance
chat_memory = ChatMemory()

# -------------------------
# Agent Response Function
# -------------------------
def agent_chat(user_message, patient_info=None):
    # Add user message to memory
    chat_memory.add("user", user_message)
    # Build context from memory and patient info
    context = chat_memory.get_context()
    if patient_info:
        context = f"Patient info: {patient_info}\n{context}"
    # Run agent
    response = agent.run(context)
    # Add agent response to memory
    chat_memory.add("assistant", response)
    return response

# -------------------------
# Initialize Agent
# -------------------------
agent_kwargs = {
    "prefix": AGENT_INSTRUCTION,
}

agent = initialize_agent(
    tools=[prediction_tool, ddg_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs=agent_kwargs,
    handle_parsing_errors=True,
)
