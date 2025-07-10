import streamlit as st
import pandas as pd
from io import StringIO
import requests
import json
import base64
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import sys
import traceback

# Q&A imports (for txt/pdf)
from backend.file_processing import extract_text_from_pdf, extract_text_from_txt
from backend.qa_pipeline import process_text, build_vectorstore, build_conversational_chain, summarize_text, get_sentiment
from backend.insight_extraction import (
    get_top_keywords, get_named_entities, detect_headings,
    sentence_sentiments, detect_table_like_sections, generate_wordcloud
)

# LLM Backend API config
API_BASE_URL = "http://localhost:8000"
SESSION_ID = "streamlit_session"

st.set_page_config(page_title="Unified Data Assistant", layout="wide")
st.title("🧠 Unified Conversational Data Assistant")

# ---------- Session State ---------- #
for key in ["chat_history", "text_data", "knowledge_base", "conversation_chain", "dataset_uploaded", "dataset_info"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "chat" in key else None

# ---------- Upload Section ---------- #
uploaded_file = st.file_uploader("📂 Upload a CSV / PDF / TXT file", type=["csv", "pdf", "txt"])

# ---------- CSV Handler: AI Data Analyst Flow ---------- #
def process_csv(file):
    df = pd.read_csv(file)
    st.session_state.dataset_uploaded = True
    st.session_state.dataset_info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "sample_data": df.head(3).to_string()
    }
    st.session_state.df = df

    # Dataset Info
    st.subheader("📊 Dataset Overview")
    st.write(f"**Shape:** {df.shape}")
    st.write("**Columns:**", ", ".join(df.columns))
    st.write("**Sample Data:**")
    st.dataframe(df.head())

    st.markdown("---")
    st.subheader("💬 Ask Questions About CSV Data")

    user_input = st.chat_input("Ask a question about your data...")

    if user_input:
        with st.spinner("Thinking..."):
            # API Payload
            payload = {
                "question": user_input,
                "session_id": SESSION_ID,
                "columns": ", ".join(df.columns),
                "sample_data": df.head(3).to_string()
            }

            response = requests.post(f"{API_BASE_URL}/query/", json=payload)

            if response.status_code != 200:
                st.error(f"❌ Backend error: {response.text}")
                return

            result = response.json()

            # Extract info
            response_text = result.get("response", "")
            code = result.get("code", "")
            output_type = result.get("output_type", "text")
            output_text = result.get("execution_output", "")
            fig = None

            # Try executing code if present
            if code:
                exec_globals = {
                    "df": df.copy(),
                    "pd": pd,
                    "px": px,
                    "go": go,
                    "plt": plt
                }
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
                    if "__NEEDS_INTERPRETATION__" in code:
                        output_text = output_text.replace("__NEEDS_INTERPRETATION__", "").strip()
                        payload = {"question": user_input, "code":code, "output":output_text}
                        response = requests.post(f"{API_BASE_URL}/interpret/", json=payload)
                        if response.status_code != 200:
                            st.error(f"Backend error: {response.text}")
                            return
                        response = response.json()
                        response_text = response.get("response","")
                        print("--------------------------------------------------------")
                        print("Output_Text", output_text)
                        print(response_text)

                except Exception as e:
                    sys.stdout = old_stdout
                    output_text = f"⚠️ Error executing code:\n{traceback.format_exc()}"
                    output_type = "text"

            # Append to chat history (dict-based)
            st.session_state.chat_history.append({
                "user": user_input,
                "response": response_text if response_text else output_text,
                "code": code,
                "type": "plotly" if output_type == "plotly" else "matplotlib" if output_type == "plot" else "text",
                "fig": fig
            })

    # Render chat history
    for item in st.session_state.chat_history:
        # Handle older (You, AI) tuples
        if isinstance(item, tuple):
            sender, message = item
            st.chat_message("user" if sender == "You" else "assistant").markdown(message)
            continue

        st.chat_message("user").markdown(item["user"])
        with st.chat_message("assistant"):
            st.markdown(item["response"])
            if item["type"] == "plotly" and item["fig"]:
                st.plotly_chart(item["fig"], use_container_width=False)
            elif item["type"] == "matplotlib" and item["fig"]:

                st.pyplot(item["fig"])
                plt.clf()
                plt.close()
            

            if item.get("code"):
                with st.expander("🔍 View Generated Code"):
                    st.code(item["code"], language="python")

# ---------- PDF/TXT Handler: Q&A Flow ---------- #
def process_textual_file(file, file_type):
    if file_type == "application/pdf":
        text = extract_text_from_pdf(file)
    else:
        text = extract_text_from_txt(file)

    st.session_state.text_data = text
    st.success("✅ File processed successfully!")

    # Show insights
    with st.expander("📊 Document Insights"):
        st.write(f"**Words:** {len(text.split())} | **Estimated Reading Time:** {round(len(text.split())/200)} min")

        sentiment, polarity = get_sentiment(text)
        st.metric("Sentiment", sentiment)
        st.metric("Polarity", round(polarity, 3))

        top_keywords = get_top_keywords(text)
        with st.expander("🔑 Top Keywords"):
            for word, count in top_keywords[:5]:
                st.markdown(f"- **{word}**: {count}")

        entities = get_named_entities(text)
        with st.expander("🧠 Named Entities"):
            for ent_type, ent_vals in entities.items():
                st.markdown(f"**{ent_type}**")
                for name, count in ent_vals:
                    st.markdown(f"- {name} ({count})")

        headings = detect_headings(text)
        if headings:
            with st.expander("📌 Headings"):
                for h in headings:
                    st.markdown(f"- {h}")

        sentiments = sentence_sentiments(text)
        with st.expander("🎭 Sentence-wise Sentiment (Top 5)"):
            for i, (sentence, score) in enumerate(sentiments[:5]):
                st.markdown(f"{i+1}. _{sentence}_ → **Polarity:** {score}")

        tables = detect_table_like_sections(text)
        if tables:
            with st.expander("📋 Table-like Sections"):
                for block in tables:
                    st.code(block, language="text")

        if st.button("🖼️ Generate Word Cloud"):
            fig = generate_wordcloud(text)
            st.pyplot(fig)

    # Summary
    if st.button("🪄 Generate Summary"):
        summary = summarize_text(text)
        st.subheader("📑 Summary")
        st.write(summary)

    # Vector store and chain
    chunks = process_text(text)
    st.session_state.knowledge_base = build_vectorstore(chunks)
    st.session_state.conversation_chain = build_conversational_chain(st.session_state.knowledge_base)

    # Chat
    st.markdown("---")
    st.subheader("💬 Ask Questions from Document")

    user_input = st.chat_input("Ask a question...")

    if user_input:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation_chain.run(user_input)

        st.session_state.chat_history.append({
            "user": user_input,
            "response": response,
            "code": None,
            "type": "text",
            "fig": None
        })

    for item in st.session_state.chat_history:
        if isinstance(item, tuple):
            # Fallback for old format
            sender, message = item
            st.chat_message("user" if sender == "You" else "assistant").markdown(message)
        else:
            st.chat_message("user").markdown(item["user"])
            st.chat_message("assistant").markdown(item["response"])

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.conversation_chain.memory.clear()
        st.rerun()


# ---------- Routing Based on File Type ---------- #
if uploaded_file:
    file_type = uploaded_file.type
    if file_type == "text/csv":
        process_csv(uploaded_file)
    elif file_type in ["application/pdf", "text/plain"]:
        process_textual_file(uploaded_file, file_type)
    else:
        st.error("❌ Unsupported file type")

# ---------- Footer ---------- #
st.markdown("---")
st.markdown("Built with ❤️ using FastAPI + Streamlit")

