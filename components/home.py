import streamlit as st

def show_home_page():
    st.title("Welcome to CodeCraft GPT!")
    st.write("This is the home page.")

    st.markdown("""
    CodeCraft GPT is an all-in-one platform designed for AI/ML developers.
    It offers a set of mini-apps for code documentation, optimization, refactoring,
    style checking, testing, language translation, and database interaction.
    """)