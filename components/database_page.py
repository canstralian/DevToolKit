# components/database_page.py
import streamlit as st
import sqlite3

def show_database_page(chat):
    st.title("Database Interaction")
    st.write("This page interacts with a SQLite database.")

    def execute_query(query):
        conn = sqlite3.connect("test.db")
        result = conn.execute(query)
        return result.fetchall()

    database_query = st.text_input("Enter the SQLite query:", "SELECT * FROM users;")

    if st.button("Execute Query"):
        result = execute_query(database_query)
        st.write(result)