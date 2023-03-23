import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    loader = PyPDFLoader("content/Treasury Management Book .pdf")
    pages = loader.load_and_split()
    index = VectorstoreIndexCreator().from_loaders([loader])

    qu = "What is treasury management"
    response = index.query(str(qu))
    st.write(qu)

except IndexError as e:
    st.write("IndexError occurred:", e)
    st.write("Creating VectorstoreIndex...")
    loader = PyPDFLoader("content/Treasury Management Book .pdf")
    pages = loader.load_and_split()
    index = VectorstoreIndexCreator().from_loaders([loader])
    
    qu = "What is treasury management"
    response = index.query(str(qu))
    st.write(qu)
