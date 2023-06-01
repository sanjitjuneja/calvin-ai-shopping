import streamlit as st
from st_pages import show_pages_from_config, add_page_title

st.set_page_config(page_title="Home", layout="centered")


# Setup App Layout
st.title("Welcome To Verbyl's AI Platform! 👋")
st.markdown(
    """ 
        > :black[**Hey There 🙋‍♂️**]
        > :black[*Use any of the following agents to help with your needs!*]
        """
)

sports, productivity, creators = st.columns(3)

with sports:
	st.image("assets/calvin.png", width=100),
	st.title("Calvin (Sports) 🏀"),
	st.markdown(
		""" 
			> :black[**Hey There, I'm Calvin 👋**]
			> :black[*👨 I'm your AI Assistant, trained specifically to answer any and all questions related to sports.*]
			> :black[*🧠 The cool thing about me is that I can remember things from all our conversations really well!*]
			"""
	)

with productivity:
	st.image("assets/calvin.png", width=100),
	st.title("Calvin (Sports) 🏀"),
	st.markdown(
		""" 
			> :black[**Hey There, I'm Calvin 👋**]
			> :black[*👨 I'm your AI Assistant, trained specifically to answer any and all questions related to sports.*]
			> :black[*🧠 The cool thing about me is that I can remember things from all our conversations really well!*]
			"""
	)


add_page_title()

show_pages_from_config()