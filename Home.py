import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import os
import openai

openai.api_key = os.getenv("API_KEY")


loader = PyPDFLoader("content/Treasury Management Book .pdf")
pages = loader.load_and_split()
index = VectorstoreIndexCreator().from_loaders([loader])



query = st.text_input("Type your message here")

if query is not None:
    response = index.query(query)
    st.write(response)