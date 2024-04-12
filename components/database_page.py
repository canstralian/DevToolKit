import streamlit as st
from langchain.chat_models import ChatOpenAI
from data.database_system import DATABASE_SYSTEM

def show_database_page(chat):
    st.title("Database Interaction")
    st.write("This page interacts with a database system.")

    database_selection = st.selectbox("Select a database system:", DATABASE_SYSTEM)
    database_query = st.text_input("Enter the database query:")

    if st.button("Execute Query"):
        # Implement the functionality for executing a query
        pass