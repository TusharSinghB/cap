from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import re
import traceback
import os
from dotenv import load_dotenv
import io
import sys
import base64
import json
from typing import Optional, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="AI Data Analyst API", version="1.0.0")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
df_storage = {}
chat_histories = {}

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    response: str
    code: str = ""
    output_type: str = "text"  # "text", "plot", "error"
    plot_data: Optional[str] = None  # base64 encoded plot or plotly JSON
    execution_output: str = ""

class DatasetInfo(BaseModel):
    shape: tuple
    columns: list
    sample_data: str

# LLM Setup
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY
    )

llm = get_llm()

# Prompts
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
- A final line: `print("__NEEDS_INTERPRETATION__")` so the system can interpret the results

---

### 🧩 3. Driver Analysis Rules

If the user intent is **driver_analysis**:
- Use a tree-based model like `RandomForestRegressor` or `RandomForestClassifier` from `sklearn`
- Encode categorical features using `LabelEncoder`
- Identify the target variable based on user query (e.g., "price", "churn", etc.)

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

prompt_2 = PromptTemplate(
    input_variables=["request", "code", "output"],
    template='''
    question : {request}
    code : {code}
    output : {output}

    Please provide a human-readable summary of this output keep it short. 
    Maintain consistent formatting and use bold to highlight important things.
    Example 1:
    """
    question: What is the average of price?
    code: df['price'].mean
    output: 24.34
    
    llm output: The average price of 24.34
    """
    '''
)

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
    
    chain_with_history = RunnableWithMessageHistory(
        prompt_1 | llm,
        get_session_history=get_session_history,
        input_messages_key="request",
        history_messages_key="chat_history"
    )
    
    return chain_with_history

def matplotlib_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    return image_base64

def plotly_to_json(fig):
    """Convert plotly figure to JSON string"""
    return fig.to_json()

@app.post("/upload-dataset/")
async def upload_dataset(file: UploadFile = File(...), session_id: str = "default"):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Store dataframe
        df_storage[session_id] = df
        
        return DatasetInfo(
            shape=df.shape,
            columns=df.columns.tolist(),
            sample_data=df.head(3).to_string()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/query/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        if request.session_id not in df_storage:
            raise HTTPException(status_code=400, detail="No dataset uploaded for this session")
        
        df = df_storage[request.session_id]
        
        # Setup chain for this session
        chain_with_history = setup_chain(request.session_id)
        
        # Get LLM response
        inputs = {
            "request": request.question,
            "columns": ", ".join(df.columns.tolist()),
            "sample_data": df.head(3).to_string()
        }
        
        result = chain_with_history.invoke(
            inputs,
            config={"configurable": {"session_id": request.session_id}}
        )
        
        llm_response = result.content
        
        # Extract code from response
        code_match = re.search(r'```python\n(.*?)```', llm_response, re.DOTALL)
        
        if not code_match:
            return QueryResponse(
                response=llm_response,
                code="",
                output_type="text",
                execution_output=""
            )
        
        generated_code = code_match.group(1).strip()
        
        # Execute code
        exec_globals = {
            'df': df,
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'px': px,
            'go': go
        }
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            exec(generated_code, exec_globals)
        except Exception as e:
            sys.stdout = old_stdout
            error_trace = traceback.format_exc()
            return QueryResponse(
                response=f"❌ Error executing code:\n```\n{error_trace}\n```",
                code=generated_code,
                output_type="error",
                execution_output=error_trace
            )
        
        sys.stdout = old_stdout
        executed_output = buffer.getvalue()
        
        # Handle different output types
        if 'plt.show()' in generated_code:
            fig = plt.gcf()
            plot_data = matplotlib_to_base64(fig)
            plt.close()
            return QueryResponse(
                response="Visualization rendered above.",
                code=generated_code,
                output_type="plot",
                plot_data=plot_data,
                execution_output=executed_output
            )
        
        elif 'fig' in exec_globals:
            fig = exec_globals['fig']
            plot_data = plotly_to_json(fig)
            return QueryResponse(
                response="Plot generated using Plotly.",
                code=generated_code,
                output_type="plotly",
                plot_data=plot_data,
                execution_output=executed_output
            )
        
        elif "__NEEDS_INTERPRETATION__" in executed_output:
            cleaned_output = executed_output.replace("__NEEDS_INTERPRETATION__", "").strip()
            chain = LLMChain(llm=llm, prompt=prompt_2)
            summary = chain.run(
                request=request.question,
                code=generated_code,
                output=cleaned_output
            )
            return QueryResponse(
                response=summary,
                code=generated_code,
                output_type="text",
                execution_output=cleaned_output
            )
        
        else:
            return QueryResponse(
                response=executed_output.strip(),
                code=generated_code,
                output_type="text",
                execution_output=executed_output
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/clear-chat/")
async def clear_chat(session_id: str = "default"):
    if session_id in chat_histories:
        chat_histories[session_id].clear()
    return {"message": "Chat history cleared"}

@app.get("/dataset-info/")
async def get_dataset_info(session_id: str = "default"):
    if session_id not in df_storage:
        raise HTTPException(status_code=404, detail="No dataset found for this session")
    
    df = df_storage[session_id]
    return DatasetInfo(
        shape=df.shape,
        columns=df.columns.tolist(),
        sample_data=df.head(5).to_string()
    )

@app.get("/health/")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)