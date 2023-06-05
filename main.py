# IMPORTS
import os
import streamlit as st
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from Auth import Auth
from Sidebar import Sidebar
from calvin_model.calvin_agi import CalvinAGI

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
auth = Auth()


# FUNCTION: DEFINE INITIAL EMBEDDINGS
def initial_embeddings(openai_api_key, first_task):
    with st.spinner("Preparing Model..."):
        # Define your embedding model
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key, model="text-embedding-ada-002"
        )

        vectorstore = FAISS.from_texts(
            ["_"], embeddings, metadatas=[{"task": first_task}]
        )
    return vectorstore



# MAIN APP FUNCTIONALITY
def main():

	# APP LAYOUT
	st.image("assets/calvin.png", width=150)
	st.title("Calvin: Your AI Shopper 🛍️")
	status_bar = st.text(" ")
	st.markdown(
		""" 
			> :black[**Hey There, I'm Calvin 👋**]
			> :black[*🛒 I'm your personal intelligent shopper, here to enhance your buying experience with AI.*]
			> :black[*🧠 Plus, I directly integrate with the web to guide you through every step of the way!*]
			"""
	)
	st.text("---")


	# SIDEBAR
	sidebar = Sidebar(auth)
	sidebar.main()


	# MAIN
	if auth.get_authentication_status():
		OBJECTIVE = """You are Calvin, an AI Intelligent Shopper who enhances people's buying experience when shopping online. You perform one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
		MAX_ITERATIONS = sidebar.get_max_iterations()
		
		user_input, submit = st.columns([7, 1])
		first_task = user_input.text_input(
			label="Ask Calvin Anything",
			key="user_input",
			placeholder="Type Here...",
			value="Find me a premium webcam good for video calls.",
			label_visibility="collapsed",
		)
		user_submit = submit.button("Submit", type="primary")
		vectorstore = initial_embeddings(os.environ["OPENAI_API_KEY"], first_task)


		if user_submit:
			if first_task == "":
				st.warning("Please Enter A Prompt")
			else:
				try:
					calvin = CalvinAGI.from_llm_and_objectives(
						llm=OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"]),
						vectorstore=vectorstore,
						objective=OBJECTIVE,
						first_task=first_task,
						verbose=False,
					)
					with st.spinner("🛍️ Calvin Working ..."):
						calvin.run(max_iterations=MAX_ITERATIONS)

				except Exception as e:
					st.error(e)


if __name__ == "__main__":
	main()