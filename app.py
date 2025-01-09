import streamlit as st
import os
from rag import EnhancedRAGQuerySystem  # Import your RAG class

def init_rag():
    api_key = st.secrets["OPENAI_API_KEY"]  
    return EnhancedRAGQuerySystem(api_key=api_key)

def main():
    st.title("Casino Application Q&A")
    
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = init_rag()
        st.session_state.rag_system.load_qa_data(["qa_output2023.json", "qa_output2024.json"])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask a question about the casino application process"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        response = st.session_state.rag_system.query(prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main()
