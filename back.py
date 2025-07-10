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
from langchain.chains import LLMChain

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
- A final line: `print("__NEEDS_INTERPRETATION__")` so the system can interpret the results


---
### 🧩 3. Driver Analysis Rules 
If the user intent is driver_analysis:
Assume the target variable is mentioned or implied in the user query (e.g., “price”, “sales”, “churn”)
Use a tree-based model: RandomForestRegressor (for numeric target) or RandomForestClassifier (for categorical)
Encode:
Categorical features using LabelEncoder
Ordinal features (if known) using OneHotEncoder
Perform analysis on copy_df, not the original df
After model training:
Extract .feature_importances_
Sort them in descending order
Print a pandas Series or DataFrame of top features with importance scores
Generate a plot using  plotly:

Labels = feature names
Values = importance scores
Include meaningful title
Add this line in code print("__NEEDS_INTERPRETATION__")



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
Example
```
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

copy_df = df.copy()
copy_df = copy_df.dropna(subset=['SALES'])

# Encode categorical variables
for col in copy_df.select_dtypes(include='object').columns:
    copy_df[col] = LabelEncoder().fit_transform(copy_df[col])

X = copy_df.drop(columns=['SALES'])
y = copy_df['SALES']

model = RandomForestRegressor()
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

print("Top Driver Features:\n", top_features)

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(top_features.values, labels=top_features.index, autopct='%1.1f%%', startangle=140)
plt.title("Top Drivers Influencing SALES")
plt.tight_layout()
plt.show()

print("__NEEDS_INTERPRETATION__")
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
You are an AI data assistant. Your task is to generate a short and readable summary for the user based on the following:

- question: {request}
- code: {code}
- output: {output}

### Instructions:
- Keep the summary **brief and human-friendly**
- Do **not** repeat the entire output unless necessary (e.g., list of values)
- Highlight important information using **bold**
- Use natural phrasing (e.g., "The average price is **24.34**")
- Avoid redundant phrases like "The output lists..." or "The result is...

Driver Driver output code in tabular format
📄 Driver Analysis Report Template
Rank | Feature | Importance Score | Interpretation
example
```
   Rank        Feature  Importance Score               Interpretation
   1     PRODUCTLINE             0.25  Interpretation for PRODUCTLINE
   2        DEALSIZE             0.20  Interpretation for DEALSIZE
   3  QUANTITYORDERED            0.15  Interpretation for QUANTITYORDERED
   4         COUNTRY             0.12  Interpretation for COUNTRY
```

### Output Format:
Return only the final user-friendly summary. Do **not** include any labels like "llm output" or examples.

### Example 1
Question: What is the average price?
Output: 24.34  
✅ Final Summary: The average price is **24.34**

### Example 2  
Question: What are the unique brands in the dataset?  
Output: Honda, Toyota, Ford, BMW  
✅ Final Summary: The dataset includes the brands: **Honda**, **Toyota**, **Ford**, and **BMW**.
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

@app.post("/interpret/")
async def interpret_output(payload: InterpretationRequest):
    chain = LLMChain(llm=llm, prompt=prompt_2)
    result = chain.run({
        "request": payload.question,
        "code": payload.code,
        "output": payload.output
    })
    return {"response": result}

# ------------------------------------------------------------------------------------------------------------
# Unstructured Data

from PyPDF2 import PdfReader
import csv
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text
def extract_text_from_txt(txt_file):
    return txt_file.read().decode('utf-8')

@app.post("/extract-pdf-text/")
async def extract_pdf_text(file: str):
    try:
        text = extract_text_from_pdf(file)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract PDF text: {str(e)}")
    
@app.post("/extract-txt/")
async def extract_txt(file:str):
    try:
        return {"text":extract_text_from_txt(file)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text document: {str(e)}")
