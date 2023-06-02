# IMPORTS
import os
import time
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import (HumanMessage, SystemMessage)
os.environ["OPENAI_API_KEY"] = "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"


# SETUP: PAGE CONFIGURATION
st.set_page_config(page_title="Oscar: Content Creation AI Bot 👨‍💻", layout="centered")


# SETUP: INITIALIZE SESSION STATES
if "oscar_generated" not in st.session_state:
    st.session_state["oscar_generated"] = []
if "oscar_past" not in st.session_state:
    st.session_state["oscar_past"] = []
if "oscar_input" not in st.session_state:
    st.session_state["oscar_input"] = ""
if "oscar_stored_session" not in st.session_state:
    st.session_state["oscar_stored_session"] = []


# SETUP: APP LAYOUT CONFIGURATION
st.image("assets/oscar.png", width=150)
st.title("Oscar: Content Creation AI Bot 👨‍💻")
if len(st.session_state["oscar_generated"]) > 0:
    status_bar = st.progress(100)
else:
    status_bar = st.text(" ")
st.markdown(
    """ 
        > :black[**Hey There, I'm Oscar 👋**]
        > :black[*👨 I'm your AI Assistant, trained specifically to answer any and all questions related to content creation.*]
        > :black[*🧠 The cool thing about me is that I can remember things from all our conversations really well!*]
        """
)
st.text("---")


# FUNCTION: GET TEXT INPUT
def oscar_get_text():
    """
    Gets text input from user.
    """
    # Get text input
    oscar_user_input = st.text_input(
        "Ask Oscar Anything:",
        st.session_state["oscar_input"],
        key="input",
        placeholder="Type Here...",
        label_visibility="hidden",
    )
    # Return input
    return oscar_user_input


# FUNCTION: START A NEW CHAT
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state["oscar_generated"]) - 1, -1, -1):
        save.append("User:" + st.session_state["oscar_past"][i])
        save.append("Bot:" + st.session_state["oscar_generated"][i])
    st.session_state["oscar_stored_session"].append(save)
    st.session_state["oscar_generated"] = []
    st.session_state["oscar_past"] = []
    st.session_state["oscar_input"] = ""


# FUNCTION: CLEAR CHAT HISTORY
def clear_history():
    del st.session_state.oscar_stored_session


# SIDEBAR: MODEL SETTINGS
with st.sidebar.expander("🧠 Model Settings ", expanded=False):
    # MODEL
    MODEL = st.selectbox(
        label="Model",
        options=[
            "gpt-3.5-turbo",
        ],
    )
    # MODEL TEMPERATURE
    TEMP = st.slider(
        "Randomness", min_value=0.0, max_value=1.0, step=0.01, value=0.5, format='%f',
    )


# SIDEBAR: NEW CHAT & CLEAR HISTORY BUTTON
st.sidebar.button("New Chat", on_click=new_chat, type="primary")
if st.session_state.oscar_stored_session:
    st.sidebar.button("Clear History", on_click=clear_history, type="secondary")


# SIDEBAR: DISPLAY STORED SESSIONS
if st.session_state.oscar_stored_session:
    for i, sublist in enumerate(st.session_state.oscar_stored_session):
        with st.sidebar.expander(label=f"Conversation {i+1}:"):
            st.write(sublist)



# MAIN: GET USER INPUT
oscar_user_input = oscar_get_text()

# MAIN: CALL MODEL
def generate_response(user_input):
    """
    Generates response from user input.
    """
    # Create Model Instance
    model = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name=MODEL,
        temperature=TEMP,
        max_tokens=150,
        stop=["\n", "User:", "Bot:"],
    )
    messages = [
        SystemMessage(content="You are Oscar, a content creation AI Assistant who helps answer any all questions related to content creation."),
        HumanMessage(content=user_input),
    ]
    # Send Response
    response = model(messages)
    # Return response
    return response.content


# MAIN: RESPONSE WORKFLOW
if oscar_user_input:
    for i in range(15):
        status_bar.progress(i, text="Sending...")
        time.sleep(0.05)
    st.text(" ")
    # Get response
    response = generate_response(oscar_user_input)
    # Update status bar
    for i in range(15, 100):
        status_bar.progress(i, text="Generating...")
        time.sleep(0.02)
    status_bar.progress(100)
    # Save response
    st.session_state["oscar_generated"].append(response)
    st.session_state["oscar_past"].append(oscar_user_input)
    oscar_user_input = None


# MAIN: DISPLAY CHAT HISTORY
if st.session_state["oscar_generated"]:
    for i in range(len(st.session_state["oscar_generated"])-1, -1, -1):
        message(st.session_state["oscar_generated"][i], key=str(i))
        message(st.session_state["oscar_past"][i], is_user=True, key=str(i)+"_user")