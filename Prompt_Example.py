import os
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st
os.environ["OPENAI_API_KEY"] = "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"
# Streamlit framework

st.title('Using Langchain to read Personal files')
input_text= st.text_input("What is the query?")

# Prompt Template
# Define the input variables for the PromptTemplate
input_variables = ['name']  # Add any other input variables as needed

# Create the PromptTemplate instance
input_prompt = PromptTemplate(
    input_variables=input_variables,
    template="Tell me about the celebrity {name}"
)

# Memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
desc_memory = ConversationBufferMemory(input_key='dob', memory_key='desc_history')




# Prompts
llm = OpenAI(temperature = 0.8)
chain = LLMChain(llm = llm, prompt= input_prompt, verbose = True, output_key= 'person', memory=person_memory)


second_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

chain2 = LLMChain(llm = llm, prompt= second_prompt, verbose = True, output_key= 'dob', memory= dob_memory)

third_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events across the world on {dob}?"
)

chain3 = LLMChain(llm = llm, prompt= third_prompt, verbose = True, output_key= 'description', memory= desc_memory)



parent_chain = SequentialChain(chains=[chain, chain2, chain3], input_variables= ['name'], verbose= True, output_variables= ['person', 'dob', 'description'])

if input_text:
    st.write(parent_chain({'name': input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(desc_memory.buffer)