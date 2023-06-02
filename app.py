import os
import streamlit as st
from st_pages import show_pages_from_config, add_page_title

st.set_page_config(page_title="Home", layout="centered")


# Setup App Layout
st.title("Welcome To Verbyl's AI Platform! 👋")
st.markdown(
    """ 
        > :black[**Hey There 🙋‍♂️**]
        > :black[*👈 Select an agent from the sidebar to help with your needs!*]
        """
)


show_pages_from_config()
os.environ["OPENAI_API_KEY"] = "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"