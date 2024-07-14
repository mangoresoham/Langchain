import requests
import streamlit as st

def get_openai_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json={'input': {'topic': input_text}}
    )
    if response.status_code == 200:
        return response.json().get('output', {}).get('content', 'No content received')
    else:
        return f"Error: {response.status_code}"

def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json={'input': {'topic': input_text}}
    )
    if response.status_code == 200:
        return response.json().get('output', 'No output received')
    else:
        return f"Error: {response.status_code}"

# Streamlit framework
st.title('Langchain Demo With LLAMA2 API')

input_text = st.text_input("Write an essay on")
input_text1 = st.text_input("Write a poem on")

if input_text:
    essay_response = get_openai_response(input_text)
    st.write(essay_response)

if input_text1:
    poem_response = get_ollama_response(input_text1)
    st.write(poem_response)
