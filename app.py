# IMPORTS
import os
import random
import time
from typing import Optional
import sqlite3
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

import yaml
from yaml.loader import SafeLoader

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
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'sidebar')



# DATABASE SETUP
conn = sqlite3.connect('data.db', check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS responses(username, response)""")
c.execute("""CREATE TABLE IF NOT EXISTS prompts(username, prompt)""")
c.execute("""CREATE TABLE IF NOT EXISTS stored_sessions(username, sessions)""")
conn.commit()



# SETUP: INITIALIZE SESSION STATES
if authentication_status:
    if "responses" not in st.session_state:
        st.session_state["responses"] = c.execute("""SELECT response FROM responses WHERE username = ?""", (username,)).fetchall()
    if "prompts" not in st.session_state:
        st.session_state["prompts"] = c.execute("""SELECT prompt FROM prompts WHERE username = ?""", (username,)).fetchall()
    if "stored_sessions" not in st.session_state:
        st.session_state["stored_sessions"] = c.execute("""SELECT sessions FROM stored_sessions WHERE username = ?""", (username,)).fetchall()
    if "disableWarning" not in st.session_state:
        st.session_state["disableWarning"] = False
    conn.close()

# SETUP: APP LAYOUT CONFIGURATION
st.image("assets/calvin.png", width=150)
st.title("Calvin: Your AI Shopper 🛍️")
if authentication_status and len(st.session_state["responses"]) > 0:
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

# FUNCTION: START A NEW CHAT
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    if st.session_state["responses"]:
        save = []
        for i in range(len(st.session_state["responses"]) - 1, -1, -1):
            save.append("User:" + str(st.session_state["prompts"][i]))
            save.append("Bot:" + str(st.session_state["responses"][i]))
        st.session_state["stored_sessions"].append(save)
        conn = sqlite3.connect('data.db', check_same_thread=False)
        c = conn.cursor()
        c.execute("""INSERT INTO stored_sessions VALUES (?, ?)""", (username, str(save)))
        c.execute("""DELETE FROM responses WHERE username = ?""", (username,))
        c.execute("""DELETE FROM prompts WHERE username = ?""", (username,))
        conn.commit()
        conn.close()
        st.session_state["responses"] = []
        st.session_state["prompts"] = []


# FUNCTION: CLEAR CHAT HISTORY
def clear_history():
    conn = sqlite3.connect('data.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("""DELETE FROM stored_sessions WHERE username = ?""", (username,))
    c.execute("""DELETE FROM responses WHERE username = ?""", (username,))
    c.execute("""DELETE FROM prompts WHERE username = ?""", (username,))
    conn.commit()
    conn.close()
    del st.session_state.stored_sessions
    del st.session_state.responses
    del st.session_state.prompts


# FUNCTION: TRY EXAMPLE
def try_example():
    rand = random.randint(0, 2)
    example_prompt = ""
    if rand == 0:
        example_prompt = "Search for the iPhone 14 Pro case with the most value."
    elif rand == 1:
        example_prompt = "Formulate a shopping cart with all designer items with a total less than $1000"
    else:
        example_prompt = "Find me a premium webcam good for video calls."
    main(example_prompt)



# SIDEBAR: AUTH CHECK
if authentication_status is False:
    st.sidebar.error('Username/password is incorrect')
elif authentication_status is None:
    st.sidebar.warning('Please enter your username and password')
    st.warning('👈 Please Use Sidebar To Login/Register To Use Calvin')


# SIDEBAR: REGISTER & FORGOT FORMS
st.sidebar.text(" ")
st.sidebar.text(" ")
if authentication_status is False or authentication_status is None:
    # REGISTER USER
    with st.sidebar.expander("📝 Register", expanded=False):
        try:
            if authenticator.register_user('Register user', preauthorization=False):
                st.success('User registered successfully')
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)
    with st.sidebar.expander("🤷‍♂️ Forgot Password/Username", expanded=False):
        # FORGOT PASSWORD
        try:
            username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
            if username_forgot_pw:
                st.success('New Temporary Password: ' + random_password)
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
            else:
                st.error('Username not found')
        except Exception as e:
            st.error(e)
        # FORGOT USERNAME
        try:
            username_forgot_username, email_forgot_username = authenticator.forgot_username('Forgot username')
            if username_forgot_username:
                st.success('Username: ' + username_forgot_username)
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
            else:
                st.error('Email not found')
        except Exception as e:
            st.error(e)


# SIDEBAR: APP INFO
if authentication_status:
    st.sidebar.header("Welcome " + name + "! 👋")
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
    st.sidebar.progress(100)


# SIDEBAR: CHAT HISTORY
if authentication_status:
    st.sidebar.header("Chat History")
    st.sidebar.button("New Chat", on_click=new_chat, type="primary")
    if st.session_state.stored_sessions:
        st.sidebar.button("Clear History", on_click=clear_history, type="secondary")
    st.sidebar.text(" ")

    # SIDEBAR: DISPLAY STORED SESSIONS
    if st.session_state.stored_sessions:
        for i in range(len(st.session_state.stored_sessions) - 1, -1, -1):
            with st.sidebar.expander(label=f"Conversation {i+1}", expanded=False):
                st.write(st.session_state.stored_sessions[i])
    st.sidebar.text(" ")
    st.sidebar.progress(100)


# SIDEBAR: ACCOUNT SETTINGS & LOGOUT
if authentication_status:
    st.sidebar.text(" ")
    authenticator.logout('✌️ Logout', 'sidebar')
    with st.sidebar.expander("⚙️ Account Settings", expanded=False):
        try:
            if authenticator.update_user_details(username, 'Update user details'):
                st.success('Entries updated successfully')
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)
        try:
            if authenticator.reset_password(username, 'Reset password'):
                st.success('Password modified successfully')
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)



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








# MAIN: SETUP CALVIN
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


prefix = """You are Calvin, an AI Intelligent Shopper who enhances people's buying experience when shopping online. You perform one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
suffix = """Question: {task}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
)




llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)
verbose = True
max_iterations: Optional[int] = 1
calvin = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, task_execution_chain=agent_executor, verbose=verbose, max_iterations=max_iterations
)




# MAIN: RUN CALVIN
def main(specific_prompt: str = None):
    if text_input == "" and specific_prompt is None:
        st.session_state["disableWarning"] = True
        return
    st.session_state["disableWarning"] = False
    # retrieve input
    cur_prompt = specific_prompt if specific_prompt is not None else text_input
    
    # Update status bar
    for i in range(30):
        status_bar.progress(i, text="Sending " + "'" + cur_prompt[0:100] + "'...")
        time.sleep(0.1)
    st.text(" ")

    # Get response
    status_bar.progress(30, text="Planning & Executing...")
    response = calvin({"objective": cur_prompt})

    # Update status bar
    for i in range(30, 100):
        status_bar.progress(i, text="Generating...")
        time.sleep(0.02)
    status_bar.progress(100)

    # Save response
    st.session_state["responses"].append(response)
    st.session_state["prompts"].append(cur_prompt)
    conn = sqlite3.connect('data.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("""INSERT INTO responses VALUES (?, ?)""", (username, str(response)))
    c.execute("""INSERT INTO prompts VALUES (?, ?)""", (username, str(cur_prompt)))
    conn.commit()
    conn.close()

st.text(st.session_state["responses"])


# SUBMIT BUTTON
if authentication_status:
    with st.form("submit_user_input", clear_on_submit=True):
        input, submit = st.columns([4, 1])
        text_input = input.text_input(
            "Ask Calvin Anything:",
            key="input",
            placeholder="Type Here...",
            label_visibility="collapsed",
        )
        submit.form_submit_button(label="Submit", on_click=main, type="primary", use_container_width=True)


# MAIN: DISPLAY EMPTY PROMPT WARNING
if authentication_status and st.session_state["disableWarning"]:
    st.warning("Please Enter A Prompt")
    st.text(" ")


# MAIN: DISPLAY CHAT HISTORY
if authentication_status and st.session_state["responses"]:
    for i in range(len(st.session_state["responses"])-1, -1, -1):
        message(st.session_state["responses"][i], key=str(i))
        message(st.session_state["prompts"][i], is_user=True, key=str(i)+"_user")