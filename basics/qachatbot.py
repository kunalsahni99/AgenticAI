import os
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Simple Langchain Chatbot", page_icon="🚀")
st.title("Simple Langchain Chatbot")
st.markdown("Ask anything! **TRUST ME**, it will answer...")

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(label="Groq API key", type="password", help="Get free API key at https://console.groq.com")
    model_name = st.selectbox(
        label="Model",
        options=[
            "llama-3.1-8b-instant",
            "meta-llama/llama-4-scout-17b-16e-instruct"
        ],
        index=0
    )

    if st.button("Clear button"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None
    
    llm = ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.7,
        streaming=True
    )

    prompt = ChatPromptTemplate([
        ("system", "You are a helpful AI assistant powered by Groq. Answer the questions clearly and concisely."),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    return chain

r_chain = get_chain(api_key, model_name)

if not r_chain:
    st.warning("Please enter your Groq API key in the sidebar to get started!")
    st.markdown("[Get your free API key here](https://console.groq.com)")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if question := st.chat_input("Ask me anything"):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in r_chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▐ ")
                
                message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {str(e)}")