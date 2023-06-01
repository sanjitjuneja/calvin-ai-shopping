# IMPORTS
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import \
    ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI


# PAGE CONFIGURATION
st.set_page_config(page_title="Calvin: Sports AI Bot 🏀", layout="centered")

# APP LAYOUT CONFIGURATION
st.image("assets/calvin.png", width=150)
st.title("Calvin: Sports AI Bot 🏀")
st.markdown(
    """ 
        > :black[**Hey There, I'm Calvin 👋**]
        > :black[*👨 I'm your AI Assistant, trained specifically to answer any and all questions related to sports.*]
        > :black[*🧠 The cool thing about me is that I can remember things from all our conversations really well!*]
        """
)

# INITIALIZE SESSION STATES
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []





# GET USER INPUT
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input(
        "You: ",
        st.session_state["input"],
        key="input",
        placeholder="Ask me anything ...",
        label_visibility="hidden",
    )
    return input_text


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




# SIDEBAR: MODEL SETTINGS
with st.sidebar.expander(" 🧠 Model Settings ", expanded=False):
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
        "Randomness", min_value=0.0, max_value=1.0, step=0.01, format='%f'
    )
    
# SIDEBAR: APP SETTINGS
with st.sidebar.expander(" 🛠️ App Settings ", expanded=False):
    # PREVIEW MEMORY STORE
    if st.checkbox("Preview memory store"):
        st.write(st.session_state.entity_memory.store)
    # PREVIEW MEMORY BUFFER
    if st.checkbox("Preview memory buffer"):
        st.write(st.session_state.entity_memory.buffer)

# SIDEBAR: NEW CHAT BUTTON
st.sidebar.button("New Chat", on_click=new_chat, type="primary")





# Session state storage would be ideal
if API_O:
    # Create an OpenAI instance
    llm = ChatOpenAI(temperature=TEMP, openai_api_key=API_O, model_name=MODEL, verbose=False)

    # Create a ConversationEntityMemory object if not already created
    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=5)

    # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=st.session_state.entity_memory,
    )
else:
    st.warning(
        "API KEY REQUIRED: Please open 'Model Settings' on the right sidebar and enter your OpenAI API KEY."
    )



# APP LOGISTICS
user_input = None


# GET USER INPUT FROM TEXT_INPUT
if API_O:
	user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)








# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.info(st.session_state["past"][i], icon="👨")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    # Can throw error - requires fix
    download_str = "\n".join(download_str)
    if download_str:
        st.download_button("Download", download_str)

# Display stored conversation sessions in the sidebar
if st.session_state.stored_session:
    for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label=f"Conversation {i+1}:"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session