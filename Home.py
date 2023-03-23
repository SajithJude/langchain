import streamlit as st

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


from langchain.document_loaders import TextLoader
loader = TextLoader('content/inputenglish.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)

query = "what is this book about"
qa.run(query)
# query = st.text_input("Type your message here")
# qu = 

# response = 
st.write(qa.run(query))
