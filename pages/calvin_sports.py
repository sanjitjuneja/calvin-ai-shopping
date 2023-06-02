# IMPORTS
import time
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import \
    ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# PAGE CONFIGURATION
st.set_page_config(page_title="Calvin: Sports AI Bot 🏀", layout="centered")

# APP LAYOUT CONFIGURATION
st.image("assets/calvin.png", width=150)
st.title("Calvin: Sports AI Bot 🏀")
status_bar = st.text(" ")
st.markdown(
    """ 
        > :black[**Hey There, I'm Calvin 👋**]
        > :black[*👨 I'm your AI Assistant, trained specifically to answer any and all questions related to sports.*]
        > :black[*🧠 The cool thing about me is that I can remember things from all our conversations really well!*]
        """
)
st.text("---")

# INITIALIZE SESSION STATES
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


# STARTS A NEW CHAT
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.entity_store = {}
    st.session_state.entity_memory.buffer.clear()

def clear_history():
    del st.session_state.stored_session



# SIDEBAR: MODEL SETTINGS
with st.sidebar.expander(" 🛠️ Model Settings ", expanded=False):
    # OpenAI API KEY
    API_O = st.text_input(
        ":blue[[OpenAI API KEY](https://openai.com/blog/openai-api)]",
    	placeholder="sk-...",
    	type="password",
	)
    # MODEL
    MODEL = st.selectbox(
        label="Model",
        options=[
            "gpt-3.5-turbo",
            "text-davinci-003",
            "text-davinci-002",
            "code-davinci-002",
        ],
    )
    # MODEL TEMPERATURE
    TEMP = st.slider(
        "Randomness", min_value=0.0, max_value=1.0, step=0.01, format='%f, default=0.5'
    )
    
# # SIDEBAR: APP SETTINGS
# with st.sidebar.expander(" 🛠️ App Settings ", expanded=False):
#     # PREVIEW MEMORY STORE
#     if st.checkbox("Preview memory store"):
#         st.write(st.session_state.entity_memory.store)
#     # PREVIEW MEMORY BUFFER
#     if st.checkbox("Preview memory buffer"):
#         st.write(st.session_state.entity_memory.buffer)

# SIDEBAR: NEW CHAT & CLEAR HISTORY BUTTON
st.sidebar.button("New Chat", on_click=new_chat, type="primary")
if st.session_state.stored_session:
    st.sidebar.button("Clear History", on_click=clear_history, type="secondary")

# SIDEBAR: DISPLAY STORED SESSIONS
if st.session_state.stored_session:
    for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label=f"Conversation {i+1}:"):
            st.write(sublist)





def get_text():
    """
    Gets text input from user.
    """
    # Get text input
    user_input = st.text_input(
        "You: ",
        st.session_state["input"],
        key="input",
        placeholder="Type Here...",
        label_visibility="hidden",
    )
    # Return input
    return user_input

def generate_response(user_input):
    """
    Generates response from user input.
    """
    # Get model
    model = ChatOpenAI(
        openai_api_key=API_O,
        model_name=MODEL,
        temperature=TEMP,
        max_tokens=150,
        stop=["\n", "User:", "Bot:"],
    )
    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=model, k=5)
    # MESSAGES
    messages = [
        SystemMessage(content="You are Calvin, a sports AI Assistant who helps answer any all questions related to sports."),
        HumanMessage(content=user_input),
    ]
    # Send Response
    response = model(messages)
    # Return response
    return response.content


if not API_O:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    user_input = None



user_input = get_text()

if user_input:
    for i in range(15):
        status_bar.progress(i, text="Sending...")
        time.sleep(0.05)
    st.text(" ")
    # Get response
    response = generate_response(user_input)
    # Update status bar
    for i in range(15, 100):
        status_bar.progress(i, text="Generating...")
        time.sleep(0.02)
    status_bar.progress(100)
    # Save response
    st.session_state["generated"].append(response)
    st.session_state["past"].append(user_input)
    user_input = None

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i)+"_user")