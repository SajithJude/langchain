import streamlit as st
import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
openai.api_key =  os.getenv("OPENAI_API_KEY ")


loader = PyPDFLoader("content/Treasury Management Book .pdf")
# data = loader.load()
index = VectorstoreIndexCreator().from_loaders([loader])

query = st.text_input("Enter your question")

if st.button("Submit"):
    respones = index.query_with_sources(query)
    st.write(respones)
    sources = respones['sources']
    st.write(sources)
    if sources[0] == "content/Treasury Management Book .pdf":
        st.write(respones['answer'])
        answer = st.write(respones['answer'])



# chain = load_qa_chain(llm, chain_type="stuff")
# chain.run(input_documents=docs, question=query)

# # from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter
# from langchain import OpenAI, VectorDBQA





# from langchain.document_loaders import TextLoader
# loader = TextLoader('content/inputenglish.txt')
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()
# docsearch = Chroma.from_documents(texts, embeddings)
# qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)

# query = st.text_input("Enter your question")
# # qa.run(query)
# # query = st.text_input("Type your message here")
# # qu = 
# if query is not None:
# # response = 
#     st.write(qa.run(query))

