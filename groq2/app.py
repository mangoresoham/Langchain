import streamlit as st
import os 
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

## load GROQ API Key

groq_api_key=os.getenv('GROQ_API_KEY')
st.title("ChatGroq With Llama3 Demo.")

llm=ChatGroq(groq_api_key=groq_api_key,
             model="Llama3-70b-8192")


prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions:{input}
"""
)

def vector_embedding():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./ML") #Data Ingestion
        st.session_state.docs=st.session_state.loader.load() #Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=500) #Document Chunks Creation
        st.session_state.final_document=st.session_state.text_splitter.split_documents(st.session_state.docs)   # Splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_document,st.session_state.embeddings) #Vector Ollama Embedding


prompt1=st.text_input("Enter your question from the document")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB is Ready")

import time 

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retriever_chain.invoke({'input':prompt1})
    print("Response Time:", time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        #Find Relevent Chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------------------------")