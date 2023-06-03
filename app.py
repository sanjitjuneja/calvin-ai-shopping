# IMPORTS
import os
import random
import time
from typing import Optional
import streamlit as st
from streamlit_chat import message
import streamlit_authenticator as stauth

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools import AIPluginTool

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.experimental import BabyAGI

# API KEYS
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "fc35ee3159ee64b6f23fa05b2083b87e6a6ccd9178f961b2e4196ce5f7b510ae"


# SETUP: PAGE CONFIGURATION
st.set_page_config(page_title="Calvin: AI Shopper", page_icon="assets/calvin.png", layout="centered", initial_sidebar_state="auto")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



# AUTHENTICATION SETUP
import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)


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
st.title("Calvin: Your AI Shopper 🛍️")
if len(st.session_state["calvin_generated"]) > 0:
    status_bar = st.progress(100)
else:
    status_bar = st.text(" ")
st.markdown(
    """ 
        > :black[**Hey There, I'm Calvin 👋**]
        > :black[*🛒 I'm your personal intelligent shopper, here to enhance your buying experience with AI.*]
        > :black[*🧠 Plus, I directly integrate with apps like Klarna to guide you through every step of the way!*]
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
        value=st.session_state["calvin_input"],
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
    if st.session_state["calvin_generated"]:
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

# FUNCTION: TRY EXAMPLE
def try_example():
    rand = random.randint(0, 2)
    exampleStr = ""
    if rand == 0:
        exampleStr = "Search for the iPhone 14 Pro case with the most value."
    elif rand == 1:
        exampleStr = "Formulate a shopping cart with all designer items with a total less than $1000"
    else:
        exampleStr = "Find me a premium webcam good for video calls."
    st.session_state["calvin_input"] = exampleStr

# TODO: Add Login/Register/Forgot Functionality, 
# Hide chat history, user input, all sidebar info until logged in

# SIDEBAR: APP INFO
st.sidebar.header("Welcome Sanjit! 👋") # TODO: Add user's name
st.sidebar.markdown(""":black[Use Calvin to help you shop!]""")
st.sidebar.button("⌛️ Try Example", on_click=try_example, type="secondary")
with st.sidebar.expander("✍️ Prompt Examples", expanded=False):
    st.markdown(
    """ 
        :black[For Best Results, Use An Objective Format:\n]
        ------
        :black[1. *"Find me a premium webcam good for video calls."*]
        :black[2. *"Search for the iPhone 14 Pro case with the most value."*]
        :black[3. *"Formulate a shopping cart with all designer items with a total less than $1000"*]
        """
    )
st.sidebar.progress(0)


# SIDEBAR: CHAT HISTORY
st.sidebar.header("Chat History")
newChatBut, clearHistoryBut = st.sidebar.columns([0.5, 0.6])
newChatBut.button("New Chat", on_click=new_chat, type="primary")
if st.session_state.calvin_stored_session:
    clearHistoryBut.button("Clear History", on_click=clear_history, type="secondary")
st.sidebar.text(" ")

# SIDEBAR: DISPLAY STORED SESSIONS
if st.session_state.calvin_stored_session:
    for i, sublist in enumerate(st.session_state.calvin_stored_session):
        with st.sidebar.expander(label=f"Conversation {i+1}:"):
            st.write(sublist)
        # TODO: Add button to re-load session data & chat in main window
# st.sidebar.progress(0)
st.sidebar.text(" ")
st.sidebar.progress(0)


# SIDEBAR: ACCOUNT INFO
logout, settings = st.sidebar.columns([0.5, 0.55])
settings.button("⚙️ Settings", type="secondary") # TODO: Add settings functionality
logout.button("✌️ Logout", type="secondary") # TODO: Add logout functionality




# MEMORY SETUP: VECTOR STORE
embeddings_model = OpenAIEmbeddings()
import faiss
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})




# TOOL SETUP: TODO_CHAIN
todo_prompt = PromptTemplate.from_template(
    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
)
todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)

# TOOL SETUP: SEARCH
search = SerpAPIWrapper()

# TOOL SETUP: KLARNA
klarna = AIPluginTool.from_plugin_url("https://www.klarna.com/.well-known/ai-plugin.json")
llm = ChatOpenAI(temperature=0)
tools = load_tools(["requests_all"])
tools += [klarna]

klarna_agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)








# MAIN: SETUP Calvin
llm = OpenAI(temperature=0)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for searching the web for the new information, product releases, product information, etc.",
    ),
    Tool(
        name="Klarna",
        func=klarna_agent_chain.run,
        description="useful for specific product recommendations, product information, creating carts for the user. Usually used in the latter steps."
    ),
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
]


prefix = """You are Calvin, an AI Intelligent Shopper who enhances people's buying experience when shopping online. You perform one task based on the following objective: {objective}."""
# suffix = """Question: {task}
# {agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix="",
    input_variables=["objective"],
)




llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=False
)
verbose = False
max_iterations: Optional[int] = 3
calvin = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, task_execution_chain=agent_executor, verbose=verbose, max_iterations=max_iterations
)

def small():
    st.session_state["calvin_input"] = calvin_user_input

# MAIN: USER INPUT
calvin_user_input = st.text_input(
    "Ask Calvin Anything:",
    st.session_state["calvin_input"],
    key="input",
    placeholder="Type Here...",
    label_visibility="hidden",
    on_change=small
)


# MAIN: PROCESS USER INPUT
if st.session_state["calvin_input"] != "":
    # Update status bar
    for i in range(15):
        status_bar.progress(i, text="Sending...")
        time.sleep(0.05)
    st.text(" ")

    # Get response
    response = "This would be Calvin's response."
    # response = calvin({"objective": calvin_user_input})

    # Update status bar
    for i in range(15, 100):
        status_bar.progress(i, text="Generating...")
        time.sleep(0.02)
    status_bar.progress(100)

    # Save response
    st.session_state["calvin_generated"].append(response)
    st.session_state["calvin_past"].append(calvin_user_input)

    # Reset user input
    st.session_state["calvin_input"] = ""




# MAIN: DISPLAY CHAT HISTORY
if st.session_state["calvin_generated"]:
    for i in range(len(st.session_state["calvin_generated"])-1, -1, -1):
        message(st.session_state["calvin_generated"][i], key=str(i))
        message(st.session_state["calvin_past"][i], is_user=True, key=str(i)+"_user")