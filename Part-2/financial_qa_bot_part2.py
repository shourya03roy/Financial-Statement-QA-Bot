import streamlit as st
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.docstore import InMemoryDocstore
from langchain_community.llms import HuggingFaceHub
import os

# Initialize Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<your_huggingface_api_token>"

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"max_new_tokens": 50},
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

# Helper Function: Extract P&L Data from PDF
def extract_pl_data(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Initialize Streamlit App
st.title("Interactive QA Bot for Financial Data")
st.markdown("Upload a PDF document containing P&L data and ask financial queries.")

# Upload PDF
uploaded_file = st.file_uploader("Upload a P&L Statement PDF", type=["pdf"])
if uploaded_file:
    # Extract and preprocess P&L data
    pl_text = extract_pl_data(uploaded_file)
    
    # Placeholder: Simulate manual P&L parsing for demonstration
    pl_data = [
        {"Metric": "Revenue from operations", "2024": "37,923", "2023": "37,441"},
        {"Metric": "Other income, net", "2024": "2,729", "2023": "671"},
        {"Metric": "Profit for the period", "2024": "7,975", "2023": "6,134"},
    ]
    df_pl = pd.DataFrame(pl_data)
    
    # Create embeddings for the P&L data
    df_pl['text'] = df_pl.apply(lambda row: f"{row['Metric']}: 2024: {row['2024']}, 2023: {row['2023']}", axis=1)
    documents = df_pl['text'].tolist()
    embeddings = embedding_model.embed_documents(documents)
    
    # Set up FAISS Index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    embeddings_array = np.array(embeddings, dtype="float32")
    index.add(embeddings_array)
    
    # Create document mappings
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    docstore = InMemoryDocstore({str(i): Document(page_content=doc) for i, doc in enumerate(documents)})
    
    # Initialize FAISS vector store
    vectorstore = FAISS(
        embedding_model,
        index,
        docstore,
        index_to_docstore_id,
    )
    
    # Initialize QA Chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    
    # Show extracted P&L data
    st.markdown("### Extracted P&L Data")
    st.write(df_pl)
    
    # Query Input
    st.markdown("### Ask Your Financial Questions")
    user_query = st.text_input("Enter your query:")
    
    if user_query:
        # Get response from QA chain
        response = qa_chain.run(user_query)
        st.markdown("### Answer")
        st.write(response)
