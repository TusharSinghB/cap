import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import base64
from PIL import Image
import io

# Configuration
API_BASE_URL = "http://localhost:8000"
SESSION_ID = "streamlit_session"

# CSS Styling
st.markdown("""
    <style>
    .fixed-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 10px 20px;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.1);
        z-index: 999;
    }
    .chat-container {
        padding-bottom: 80px;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health/")
        return response.status_code == 200
    except:
        return False

def upload_dataset(file_content, filename):
    """Upload dataset to backend"""
    try:
        files = {"file": (filename, file_content, "text/csv")}
        params = {"session_id": SESSION_ID}
        response = requests.post(f"{API_BASE_URL}/upload-dataset/", files=files, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error uploading dataset: {str(e)}")
        return None

def send_query(question):
    """Send query to backend"""
    try:
        payload = {
            "question": question,
            "session_id": SESSION_ID
        }
        response = requests.post(f"{API_BASE_URL}/query/", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending query: {str(e)}")
        return None

def clear_chat():
    """Clear chat history"""
    try:
        params = {"session_id": SESSION_ID}
        response = requests.post(f"{API_BASE_URL}/clear-chat/", params=params)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error clearing chat: {str(e)}")
        return False

def get_dataset_info():
    """Get dataset information"""
    try:
        params = {"session_id": SESSION_ID}
        response = requests.get(f"{API_BASE_URL}/dataset-info/", params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None

def display_plot(plot_data, output_type):
    """Display plot based on type"""
    if output_type == "plot":
        # Matplotlib plot (base64 encoded)
        image_data = base64.b64decode(plot_data)
        image = Image.open(io.BytesIO(image_data))
        st.image(image, use_column_width=True)
    elif output_type == "plotly":
        # Plotly plot (JSON)
        fig_dict = json.loads(plot_data)
        fig = go.Figure(fig_dict)
        st.plotly_chart(fig, use_container_width=True)

def handle_send():
    """Handle send button click"""
    user_input = st.session_state.user_input_key.strip()
    if not user_input:
        return

    # Add user message to chat
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    # Send query to backend
    with st.spinner("Processing your query..."):
        result = send_query(user_input)

    if result:
        # Add assistant response to chat
        assistant_msg = {
            "role": "assistant",
            "content": result["response"],
            "code": result.get("code", ""),
            "output_type": result.get("output_type", "text"),
            "plot_data": result.get("plot_data"),
            "execution_output": result.get("execution_output", "")
        }
        st.session_state.chat_history.append(assistant_msg)

    # Clear input
    st.session_state.user_input_key = ""

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "dataset_uploaded" not in st.session_state:
    st.session_state.dataset_uploaded = False

if "dataset_info" not in st.session_state:
    st.session_state.dataset_info = None

# Main App
st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="wide")
st.title("🤖 AI Data Analyst")

# Check backend health
if not check_backend_health():
    st.error("❌ Backend server is not running. Please start the FastAPI server on port 8000.")
    st.stop()

# Sidebar for file upload
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None and not st.session_state.dataset_uploaded:
    with st.spinner("Uploading dataset..."):
        dataset_info = upload_dataset(uploaded_file.getvalue(), uploaded_file.name)
    
    if dataset_info:
        st.session_state.dataset_uploaded = True
        st.session_state.dataset_info = dataset_info
        st.sidebar.success("✅ Dataset uploaded successfully!")

# Display dataset information
if st.session_state.dataset_uploaded and st.session_state.dataset_info:
    st.subheader("📊 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Shape", f"{st.session_state.dataset_info['shape'][0]} rows × {st.session_state.dataset_info['shape'][1]} columns")
    
    with col2:
        st.metric("Columns", len(st.session_state.dataset_info['columns']))
    
    st.subheader("🔍 Column Names")
    st.write(", ".join(st.session_state.dataset_info['columns']))
    
    st.subheader("📋 Sample Data")
    st.text(st.session_state.dataset_info['sample_data'])
    
    st.markdown("---")
    st.subheader("💬 Chat with Your Data")
    
    # Chat interface
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    # Display plot if available
                    if msg.get("output_type") in ["plot", "plotly"] and msg.get("plot_data"):
                        display_plot(msg["plot_data"], msg["output_type"])
                    
                    # Display response
                    st.markdown(msg["content"])
                    
                    # Show code if available (in expander)
                    if msg.get("code"):
                        with st.expander("View Generated Code"):
                            st.code(msg["code"], language="python")
    
    # Input area
    st.text_input(
        "Ask anything about your data:",
        key="user_input_key",
        placeholder="e.g., What's the average price? Show me a bar chart of sales by region..."
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.button("Send", on_click=handle_send, type="primary")
    
    # Sidebar clear button
    if st.sidebar.button("🗑️ Clear Chat"):
        if clear_chat():
            st.session_state.chat_history = []
            st.sidebar.success("Chat cleared!")
        else:
            st.sidebar.error("Failed to clear chat")

else:
    st.info("👆 Please upload a CSV file to begin analyzing your data.")
    
    # Show example queries
    st.subheader("💡 Example Queries You Can Ask")
    
    examples = [
        "What's the average price?",
        "Show me a bar chart of sales by category",
        "What are the top 5 products by revenue?",
        "Create a scatter plot of price vs quantity",
        "What drives the highest sales?",
        "Show me the distribution of customer ages",
        "Which region has the most customers?"
    ]
    
    for example in examples:
        st.markdown(f"• {example}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using FastAPI and Streamlit")

# Additional sidebar info
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ How to Use")
st.sidebar.markdown("""
1. Upload a CSV file
2. Ask questions about your data
3. Get instant insights and visualizations
4. View the generated code
5. Clear chat when needed
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Features")
st.sidebar.markdown("""
- **Natural Language Queries**: Ask questions in plain English
- **Automatic Visualizations**: Get charts and graphs automatically
- **Code Generation**: See the Python code behind each analysis
- **Chat History**: Keep track of your analysis journey
- **Multiple Chart Types**: Bar charts, scatter plots, histograms, and more
""")