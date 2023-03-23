import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import os
import openai
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()


openai.api_key = os.getenv("OPENAI_API_KEY")


loader = PyPDFLoader("content/Treasury Management Book .pdf")
# pages = loader.load_and_split()
index = VectorstoreIndexCreator(vectorstore_cls=Chroma, 
    embedding=OpenAIEmbeddings()).from_loaders([loader])



qu = "What is treasury management"

response = index.query(str(qu))
st.write(qu)
