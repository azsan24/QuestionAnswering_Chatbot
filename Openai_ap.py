import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os


load_dotenv()




prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "Question: {question}")
    ]
)



def generate_response(question, api_key):
    """
    Generates an AI response using OpenAI's gpt-4o model.
    """
    if not api_key:
        return "Please enter your OpenAI API Key in the sidebar."

    try:
        
        llm = ChatOpenAI(
            model="gpt-4o",  
            temperature=0.7, 
            max_tokens=250,  
            openai_api_key=api_key
        )
        
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        return f"An error occurred: {e}. Check your API key and connection."

# --- Streamlit Application ---

st.set_page_config(page_title="Simple Q&A Chatbot", layout="centered")

st.title("💡 Simple Q&A Chatbot with OpenAI")

# Sidebar for API Key input
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
st.sidebar.info("Get your API key from [OpenAI Platform](https://platform.openai.com/account/api-keys).")

st.write("Go ahead, ask me anything!")

user_input = st.text_input("Your Question:", placeholder="Type your question here...")

# Button to get the answer
if st.button("Get Answer"):
    if user_input:
        with st.spinner("Thinking..."):
            response = generate_response(user_input, api_key)
        st.info(f"**Bot's Answer:**\n{response}")
    else:
        st.warning("Please type a question to get an answer.")

st.markdown("---")
st.caption("This chatbot uses OpenAI's models via LangChain.")