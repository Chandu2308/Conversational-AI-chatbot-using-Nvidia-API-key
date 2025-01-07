import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["NVIDIA_API_KEY"] = "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"  

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

st.set_page_config(page_title="Simple Chatbot", layout="wide")
st.title("HELLO..Lets Talk")

# Session state to keep track of chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages (chat history)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Define the chat prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}")
])

# Create the chain with the model and prompt
chain = prompt_template | llm | StrOutputParser()

# Input field for the user
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Append the user message to the session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user input on the chat
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Create augmented input for the model (can add context if needed)
        augmented_user_input = f"Question: {user_input}\n"

        # Stream the response from the model
        for response in chain.stream({"input": augmented_user_input}):
            full_response += response
            message_placeholder.markdown(full_response + "▌")  # Show partial response
        message_placeholder.markdown(full_response)

    # Append assistant's response to the session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})