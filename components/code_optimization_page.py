# components/code_optimization_page.py
import streamlit as st
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate)
from langchain.chat_models import ChatOpenAI
from data.optimization_prompt import OPTIMIZATION_PROMPT

def show_optimize_page(chat):
    # Request a code snippet for optimization
    st.subheader("Request Code Snippet for Optimization")
    user_code = st.text_area("Enter the code snippet:", "def factorial(n):\n\treturn 1 if n < 2 else n * factorial(n - 1)", height=200)

    if st.button("Optimize Code"):
        # Create a prompt for optimization
        optimization_prompt = ChatPromptTemplate.from_template(OPTIMIZATION_PROMPT)

        # Create a message template for the user's code
        human_message_prompt = HumanMessagePromptTemplate.from_template("{code_snippet}")

        # Combine the system and user message templates
        chat_prompt = ChatPromptTemplate.from_messages([optimization_prompt, human_message_prompt])

        # Run the optimization chat chain
        optimization_chain = LLMChain(llm=chat, prompt=chat_prompt)
        optimized_code = optimization_chain.run(code_snippet=user_code)

        # Display the optimized code
        st.subheader("Optimized Code")
        st.text_area("", optimized_code, height=200)