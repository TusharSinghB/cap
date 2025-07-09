import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import io
import re
import sys
import traceback

# Backend endpoint
API_URL = "http://localhost:8000/query/"
SESSION_ID = "frontend_session"

# Page settings
st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("🤖 AI Data Analyst")

# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

# File uploader
st.sidebar.header("📁 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df

    st.sidebar.success("✅ Dataset loaded")
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head())

    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", list(df.columns))

    st.markdown("---")
    st.subheader("💬 Chat with Your Data")

    # Chat interface
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(msg["user"])
        with st.chat_message("assistant"):
            if msg["type"] == "plotly":
                st.plotly_chart(msg["fig"], use_container_width=True)
            elif msg["type"] == "matplotlib":
                st.pyplot(msg["fig"])
            st.markdown(msg["response"])
            if msg.get("code"):
                with st.expander("View Code"):
                    st.code(msg["code"], language="python")

    # Text input for question
    user_input = st.text_input("Ask your question:", key="user_input_key")

    if st.button("Send") and user_input.strip():
        question = user_input.strip()
        df = st.session_state.df

        # Send to backend
        with st.spinner("Processing..."):
            payload = {
                "question": question,
                "columns": ", ".join(df.columns),
                "sample_data": df.head(3).to_string(),
                "session_id": SESSION_ID
            }

            try:
                res = requests.post(API_URL, json=payload)
                res.raise_for_status()
                result = res.json()

                code = result.get("code", "")
                output_type = result.get("output_type", "text")
                response_text = result.get("response", "")

                fig = None
                output_text = ""
                exec_globals = {
                    "df": df.copy(),
                    "pd": pd,
                    "plt": plt,
                    "sns": sns,
                    "px": px,
                    "go": go,
                    "st": st
                }

                if code:
                    old_stdout = sys.stdout
                    sys.stdout = buffer = io.StringIO()
                    try:
                        exec(code, exec_globals)
                        output_text = buffer.getvalue()
                        sys.stdout = old_stdout

                        if output_type == "plotly" and "fig" in exec_globals:
                            fig = exec_globals["fig"]
                        elif output_type == "plot":
                            fig = plt.gcf()

                    except Exception as e:
                        sys.stdout = old_stdout
                        output_text = f"⚠️ Error executing code:\n{traceback.format_exc()}"
                        output_type = "text"

                # Save to chat history
                st.session_state.chat_history.append({
                    "user": question,
                    "response": response_text if response_text else output_text,
                    "code": code,
                    "type": "plotly" if output_type == "plotly" else "matplotlib" if output_type == "plot" else "text",
                    "fig": fig
                })


            except requests.exceptions.RequestException as e:
                st.error(f"❌ Request failed: {e}")

# If no file yet
else:
    st.info("👆 Please upload a CSV file to start chatting with your data.")

# Clear chat
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat cleared")
