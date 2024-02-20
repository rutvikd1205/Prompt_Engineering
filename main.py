import os
from langchain_openai import OpenAI

import streamlit as st
os.environ["OPENAI_API_KEY"] = ""
# Streamlit framework

st.title('Using Langchain to read Personal files')
input_text= st.text_input("What is the query?")

# Setting up LLM model

llm = OpenAI(temperature = 0.8)

if input_text:
    st.write(llm(input_text))