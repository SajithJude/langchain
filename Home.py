import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import VectorDBQA
from langchain.llms import OpenAI
import os
import openai


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma



openai.api_key = os.getenv("OPENAI_API_KEY")


loader = PyPDFLoader("content/Treasury Management Book .pdf")
# pages = loader.load_and_split()
# index = VectorstoreIndexCreator().from_loaders([loader])



embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(loader, embeddings)


qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=db)
query = "What did the president say about Ketanji Brown Jackson"
qa.run(query)
