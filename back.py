from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# === Load environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === App config
app = FastAPI(title="AI Data Analyst API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === In-memory chat histories
chat_histories = {}

# === Models
class QueryRequest(BaseModel):
    question: str
    columns: str
    sample_data: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    response: str
    code: str
    output_type: str

# === LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)

# === PromptTemplates — paste these locally in your backend file
prompt_1 = PromptTemplate(
                input_variables=["request", "columns", "sample_data"],
                template='''You are an intelligent Python data assistant.

You are given:
- request : {request}
- columns : {columns}
- Sample rows : {sample_data}

Your job is to classify the intent and generate appropriate Python code or skip code generation depending on the instruction.

---

### 🔍 1. Intent Classification

Categorize the user request into one of the following:
- **"question"**: e.g., "What is the average price?", "Which brand has the most products?"
- **"analysis"**: e.g., grouping, correlation, filtering, statistical summaries
- **"visualization"**: e.g., bar chart, histogram, scatter plot
- **"driver_analysis"**: e.g., "What drives sales?", "Top features influencing price", "Which variables affect churn?"
- **"casual"**: e.g., greetings, thanks, jokes, or inputs not related to the dataset

---

### 🧠 2. Code Generation Rules

- Use the existing pandas DataFrame named `df` — no need to import or redefine it
- Use standard pandas operations or `scikit-learn` for model-based tasks
- Handle missing values using `.dropna()`, `fillna()`, etc.
- If the request is for **summary**, **insights**, or **brief explanation**, include:
- Proper `print()` statements for all outputs

---

### 🧩 3. Driver Analysis Rules

If the user intent is **driver_analysis**:
- Use a tree-based model like `RandomForestRegressor` or `RandomForestClassifier` from `sklearn`
- Encode categorical features using `LabelEncoder`
- Identify the target variable based on user query (e.g., “price”, “churn”, etc.)

---
📊 4. Visualization Rules (Updated)
Use matplotlib.pyplot (plt) or plotly.express (px)
For matplotlib: 
Use plt.figure(figsize=(10, 6))
Always set plt.xlabel() and plt.ylabel() with meaningful labels
Always call plt.xticks(rotation=45) if x labels are categorical and may overlap
Always use plt.tight_layout() before plt.show() to prevent label cutoff
For plotly:
Assign the chart to a variable fig
Set fig.update_layout(xaxis_title=..., yaxis_title=...) explicitly
Ensure xaxis.tickmode='linear' or xaxis.type='category' is set for categorical x-values if needed
Do not print anything or include __NEEDS_INTERPRETATION__ for visualizations

Example of Code for Visualisation
```python
import plotly.express as px

avg_price_by_model = df.groupby('car_model')['price'].mean().reset_index()
fig = px.bar(avg_price_by_model, x='car_model', y='price', title='Average Price by Car Model')

# Ensure axis titles are set
fig.update_layout(
    xaxis_title="Car Model",
    yaxis_title="Average Price",
    xaxis=dict(type='category')
)

```
---

### 💬 5. Handling Casual Requests

If the user's message is casual or unrelated to the data:
- if it is a greeting greet them if user is asking queries regarding question they ask then respond normally otherwise
- Respond like a helpful assistant if user is asking anthing else than dataset show this message "🧠 I'm your data assistant, here to help with your dataset!
It looks like your question isn't related to the uploaded data.
Try asking about trends, summaries, or insights based on your CSV file. 😊"
- Do **not** include any code
- Example:
  - User: Hey
    Response: Hello! How can I assist you with your data today?
  - User: Thank you
    Response: You're welcome! Let me know if you have more questions.

---

### 🔐 6. Code Format Requirement

Always respond with a single valid Python code block:
ALways create a copy of df named as copy_df and perform operation on copy_df
````python
<your generated code here>

'''
            )

# === Chat memory setup
def get_session_history(session_id: str):

    if session_id not in chat_histories:
        chat_histories[session_id] = StreamlitChatMessageHistory()
    return chat_histories[session_id]

def setup_chain(session_id: str):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=get_session_history(session_id)
    )
    return RunnableWithMessageHistory(
        prompt_1 | llm,
        get_session_history=get_session_history,
        input_messages_key="request",
        history_messages_key="chat_history"
    )

# === Endpoints

@app.post("/query/", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        chain = setup_chain(request.session_id)

        inputs = {
            "request": request.question,
            "columns": request.columns,
            "sample_data": request.sample_data
        }

        result = chain.invoke(
            inputs,
            config={"configurable": {"session_id": request.session_id}}
        )

        llm_response = result.content

        # Extract code
        code_match = re.search(r'```python\n(.*?)```', llm_response, re.DOTALL)
        code = code_match.group(1).strip() if code_match else ""

        # Determine output type
        if "px." in code or "plotly" in code:
            output_type = "plotly"
        elif "plt." in code:
            output_type = "plot"
        elif "RandomForest" in code:
            output_type = "driver_analysis"
        elif code:
            output_type = "code"
        else:
            output_type = "text"

        return QueryResponse(
            response=llm_response if not code else "",
            code=code,
            output_type=output_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/clear-chat/")
async def clear_chat(session_id: str = "default"):
    if session_id in chat_histories:
        chat_histories[session_id].clear()
    return {"message": "Chat cleared"}

@app.get("/health/")
async def health_check():
    return {"status": "ok"}

class InterpretationRequest(BaseModel):
    question: str
    code: str
    output: str

# @app.post("/interpret/")
# async def interpret_output(payload: InterpretationRequest):
#     chain = LLMChain(llm=llm, prompt=prompt_2)
#     result = chain.run({
#         "request": payload.question,
#         "code": payload.code,
#         "output": payload.output
#     })
#     return {"response": result}

