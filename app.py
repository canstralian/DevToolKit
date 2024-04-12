# app.py
import streamlit as st
import os

# Import components and pages
from components.optimize_page import optimize_page
    lang_page, code_documentation_page, database_page, optimize_page,
    model_trainer_page, notebook_integration_page, xai_page,
    versioning_tracker_page, nlp_nlg_page, devops_page, api_doc_page,
    code_review_page, version_control_page, recommendation_system_page,
    code_security_scanner_page, code_diagram_page
# Initialize the app
st.set_page_config(
    page_title="Codecrafter GPT: A Comprehensive Code Enhancement Platform",
    page_icon="🚀",
    layout="wide"
)

# Initialize the sidebar
st.sidebar.title("OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type='password')

if not api_key:
    st.warning("Please enter your OpenAI API key to access pages.")
else:
    # Instantiate the ChatOpenAI object
    from langchain.chat_models import ChatOpenAI
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        api_key=api_key
    )

    # Show the navigation menu
    selected = st.sidebar.select_icon("fa-icon",
        ["Home", "RefactorRite", "StyleSculpt", "TestGenius", "LangLink",
        "CodeDocGenius", "Database", "AutoOptimizer", "ModelTrainer",
        "NotebookIntegration", "ExplainableAI", "DataVersioning",
        "NLPandNLG", "DevOps", "APIDocGen", "CodeReviewAssistant",
        "VersionControl", "RecommendationSystem", "CodeSecurityScanner",
        "CodeDiagramConverter"],
        default_index=0
    )

    # Define a dictionary mapping page names to their corresponding functions
    pages = {
        "Home": home.show_home_page,
        "RefactorRite": refactor_page.show_refactor_page,
        "StyleSculpt": style_page.show_style_page,
        "TestGenius": test_page.show_test_page,
        "LangLink": lang_page.show_lang_page,
        "CodeDocGenius": code_documentation_page.show_doc_page,
        "Database": database_page.show_database_page,
        "AutoOptimizer": optimize_page.show_optimize_page,
        "ModelTrainer": model_trainer_page.show_model_trainer_page,
        "NotebookIntegration": notebook_integration_page.show_notebook_integration_page,
        "ExplainableAI": xai_page.show_xai_page,
        "DataVersioning": versioning_tracker_page.show_versioning_page,
        "NLPandNLG": nlp_nlg_page.show_nlp_nlg_page,
        "DevOps": devops_page.show_devops_page,
        "APIDocGen": api_doc_page.show_api_doc_page,
        "CodeReviewAssistant": code_review_page.show_code_review_page,
        "VersionControl": version_control_page.show_version_control_page,
        "RecommendationSystem": recommendation_system_page.show_recommendation_system_page,
        "CodeSecurityScanner": code_security_scanner_page.show_code_security_page,
        "CodeDiagramConverter": code_diagram_page.show_code_diagram_page
    }

    # Call the function for the selected page
    if selected in pages:
        pages[selected](chat)
    else:
        st.error("Page not found!")