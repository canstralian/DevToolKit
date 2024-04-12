# components/code_documentation_page.py
import streamlit as st
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

def show_doc_page(chat):
    st.title("Code Documentation Generator")
    st.write("This page generates documentation for a given code snippet.")

    code_snippet = st.text_area("Enter the code snippet:", height=200)

    if st.button("Generate Documentation"):
        # Implement the functionality for generating documentation
        system_message_prompt = SystemMessagePromptTemplate.from_template("Generate documentation for the provided code snippet.")
        human_message_prompt = HumanMessagePromptTemplate.from_template("Code snippet:\n{code_snippet}")
        chat_prompt = SystemMessagePromptTemplate.from_prompt_template(system_message_prompt)
        chat_prompt.add_message(human_message_prompt)
        llm = OpenAI(temperature=0.5)
        chain = LLMChain(llm=llm, prompt=chat_prompt)
        result = chain.run(code_snippet=code_snippet)
        st.write(result)