# IMPORTS
import os
import time
import streamlit as st
from streamlit_chat import message
import streamlit_authenticator as stauth
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools import AIPluginTool

os.environ["OPENAI_API_KEY"] = "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"


# SETUP: PAGE CONFIGURATION
st.set_page_config(page_title="Home", layout="centered")
# show_pages_from_config()


# SETUP: INITIALIZE SESSION STATES
if "calvin_generated" not in st.session_state:
    st.session_state["calvin_generated"] = []
if "calvin_past" not in st.session_state:
    st.session_state["calvin_past"] = []
if "calvin_input" not in st.session_state:
    st.session_state["calvin_input"] = ""
if "calvin_stored_session" not in st.session_state:
    st.session_state["calvin_stored_session"] = []


# SETUP: APP LAYOUT CONFIGURATION
st.image("assets/calvin.png", width=150)
st.title("Calvin: Your Intelligent Shopper 🛍️")
if len(st.session_state["calvin_generated"]) > 0:
    status_bar = st.progress(100)
else:
    status_bar = st.text(" ")
st.markdown(
    """ 
        > :black[**Hey There, I'm Calvin 👋**]
        > :black[*🛒 I'm your personal intelligent shopper, here to enhance your buying experience through AI.*]
        > :black[*🧠 Plus, I can directly integrate with various plugins to guide you through every step of the way!*]
        """
)
st.text("---")


# FUNCTION: GET TEXT INPUT
def calvin_get_text():
    """
    Gets text input from user.
    """
    # Get text input
    user_input = st.text_input(
        "Ask Calvin Anything:",
        st.session_state["calvin_input"],
        key="input",
        placeholder="Type Here...",
        label_visibility="hidden",
    )
    # Return input
    return user_input


# FUNCTION: START A NEW CHAT
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state["calvin_generated"]) - 1, -1, -1):
        save.append("User:" + st.session_state["calvin_past"][i])
        save.append("Bot:" + st.session_state["calvin_generated"][i])
    st.session_state["calvin_stored_session"].append(save)
    st.session_state["calvin_generated"] = []
    st.session_state["calvin_past"] = []
    st.session_state["calvin_input"] = ""


# FUNCTION: CLEAR CHAT HISTORY
def clear_history():
    del st.session_state.calvin_stored_session


# SIDEBAR: MODEL SETTINGS
st.sidebar.progress(0)
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

# # SIDEBAR: APP SETTINGS
# with st.sidebar.expander(" 🛠️ App Settings ", expanded=False):
#     st.text("Settings")


# SIDEBAR: NEW CHAT & CLEAR HISTORY BUTTON
st.sidebar.button("New Chat", on_click=new_chat, type="primary")
if st.session_state.calvin_stored_session:
    st.sidebar.button("Clear History", on_click=clear_history, type="secondary")


# SIDEBAR: DISPLAY STORED SESSIONS
if st.session_state.calvin_stored_session:
    for i, sublist in enumerate(st.session_state.calvin_stored_session):
        with st.sidebar.expander(label=f"Conversation {i+1}:"):
            st.write(sublist)



# MAIN: GET USER INPUT
calvin_user_input = calvin_get_text()


search = SerpAPIWrapper()











# MAIN: DISPLAY CHAT HISTORY
if st.session_state["calvin_generated"]:
    for i in range(len(st.session_state["calvin_generated"])-1, -1, -1):
        message(st.session_state["calvin_generated"][i], key=str(i))
        message(st.session_state["calvin_past"][i], is_user=True, key=str(i)+"_user")