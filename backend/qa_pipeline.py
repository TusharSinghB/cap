import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain  # for single queries
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain  # for chat-based chain
from langchain_groq import ChatGroq
from textblob import TextBlob

load_dotenv()

def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.from_texts(chunks, embeddings)

# OLD: for single question answering

def answer_question(knowledge_base, question):
    docs = knowledge_base.similarity_search(question)
    llm = ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),  # Fetch correctly from environment
        model_name='llama3-70b-8192'
    )
    chain = load_qa_chain(llm, chain_type='stuff')
    return chain.run(input_documents=docs, question=question)

# ------------------ NEW: chat-history ------------------

# Add a persistent memory object
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def build_conversational_chain(knowledge_base):
    llm = ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),
        model_name='llama3-70b-8192'
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=knowledge_base.as_retriever(),
        memory=memory
    )
# ------------------ summarize_text ------------------

def summarize_text(text):
    print(os.getenv('GROQ_API_KEY'))
    llm = ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),  # Correct environment fetching
        model_name='llama3-8b-8192'
    )
    prompt = f"Summarize the following document in 5-7 lines, highlighting key points in simple language:\n\n{text[:5000]}"
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else response  # Safety for your Groq wrapper

# -------------------EMOJI-------------------------

def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        sentiment = "😊 Positive"
    elif polarity < -0.1:
        sentiment = "😞 Negative"
    else:
        sentiment = "😐 Neutral"

    return sentiment, polarity
