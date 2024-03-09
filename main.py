from tempfile import NamedTemporaryFile

import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool

# Initialize Agent
tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key = 'chat_history',
    k=5,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key = None, # Enter your api key
    temperature=0,
    model_name = "gpt-3.5-turbo"
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)

# Set title
st.title('Ask a question to an image')

# Set header
st.header('Please upload an image')

# Upload file
file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    # Display image
    st.image(file, use_column_width=True)

    # Input text
    user_question = st.text_input('Ask a question about your image')

    # Compute agent response
    with NamedTemporaryFile(dir='.') as f:
        f.write(file.getbuffer())
        image_path = f.name
        response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))

    # Write agent response
    if user_question and user_question != "":
        st.write(response)

