## Step 1 Importing lib & loading variables 
from langchain_groq import ChatGroq
import streamlit as st 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


from dotenv import load_dotenv
load_dotenv()
import os
from langchain.prompts import ChatPromptTemplate
## Groq api 
groq_api_key=os.getenv('GROQ_API_KEY')


## Innitialize vairables

st.title('Chatgroq demo with Llama3 Demo')
## initialize the model
llm=ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192')\

## define the prompt 
prompt= ChatPromptTemplate.from_template(
    '''
    Answer the questions based on the provided context only.
    Please provide the most accuracte response based on the question.
    <context>
    {context}
    </context>
    Question={input}
    '''
)
## Def embedings, loader,finaldocs, vectors 
def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings= OllamaEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader('./us census')
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:4])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

## define prompt 
prompt1=st.text_input('Enter your Question from the documents')

if st.button('Document Embeddings'):
    vector_embeddings()
    st.write('Vector store DB is ready')

# define the chain
import time 
## define retrieval chain

# invoke the chain with expander and context and the response also
if prompt1:
      
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print('Response time:',time.process_time()-start)
    st.write(response['answer'])

    with st.expander('Documents similarity search'):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('-------------------------------')




