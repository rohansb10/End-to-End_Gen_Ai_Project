from flask import Flask, request, jsonify, render_template
from langchain_groq import ChatGroq
from langchain.document_loaders import NotebookLoader, TextLoader, WebBaseLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
import pandas as pd 
import random
import string
import textwrap
import time

import os
os.environ["LANGCHAIN_TRACING_V2"]='true'
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]="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"
os.environ["LANGCHAIN_PROJECT"]="Question Answering with Web or File"

def clean_text(pro):
    lines = pro.replace("\\n", "\n").split("\n")
    wrapped_text = "\n".join([textwrap.fill(line, width=120) for line in lines])
    return wrapped_text  # Return the cleaned and wrapped text

app = Flask(__name__)

model = 'llama3-70b-8192'
llm = ChatGroq(temperature=0.7, groq_api_key="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", model_name=model)

# Functions for different functionalities
def run_retrieval_qa_text(url, query):
    try:
        loader = TextLoader(url)
        loaded_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(loaded_documents)
        embeddings = GPT4AllEmbeddings()
        vector_store = FAISS.from_documents(texts, embeddings)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
        result = qa.run(query)
        return clean_text(result)
    except Exception as e:
        return f"An error occurred: {str(e)}"

def run_retrieval_qa_pdf(url, query):
    try:
        loader = PyPDFLoader(url)
        loaded_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(loaded_documents)
        embeddings = GPT4AllEmbeddings()
        vector_store = FAISS.from_documents(texts, embeddings)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
        result = qa.run(query)
        return clean_text(result)
    except Exception as e:
        return f"An error occurred: {str(e)}"

def run_sql_query(file, query):
    try:
        df = pd.read_csv(file)
        random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=7))
        url = f"sqlite:///{random_name}.db"
        engine = create_engine(url)
        df.to_sql(f"Your", engine, index=False)
        db = SQLDatabase(engine=engine)
        memory = ConversationBufferMemory(memory_key="chat_history")
        agent_executor = create_sql_agent(llm, db=db, memory=memory, agent_type="openai-tools", verbose=True)
        response = agent_executor.invoke({"input": query})
        return response['output']
    except Exception as e:
        return f"An error occurred: {str(e)}"

def excel(file, query):
    try:
        df = pd.read_excel(file)
        random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=7))
        url = f"sqlite:///{random_name}.db"
        engine = create_engine(url)
        df.to_sql(f"Your", engine, index=False)
        db = SQLDatabase(engine=engine)
        memory = ConversationBufferMemory(memory_key="chat_history")
        agent_executor = create_sql_agent(llm, db=db, memory=memory, agent_type="openai-tools", verbose=True)
        response = agent_executor.invoke({"input": query})
        return response['output']
    except Exception as e:
        return f"An error occurred: {str(e)}"

def run_retrieval_qa_web(url, query):
    try:
        loader = WebBaseLoader(url)
        loaded_documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(loaded_documents)
        embeddings = GPT4AllEmbeddings()
        vector_store = FAISS.from_documents(texts, embeddings)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
        result = qa.run(query)
        return clean_text(result)
    except Exception as e:
        return f"An error occurred: {str(e)}"

def jupyter_notebook(link, query=None):
    try:
        loader = NotebookLoader(link, include_outputs=True, max_output_length=20, remove_newline=True)
        loaded_documents = loader.load()
    except Exception as e:
        return f"Error loading notebook: {str(e)}"

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(loaded_documents)
    except Exception as e:
        return f"Error splitting text: {str(e)}"

    try:
        embeddings = GPT4AllEmbeddings()
        vector_store = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        return f"Error creating vector store: {str(e)}"

    try:
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
        result = qa.run(query)
    except Exception as e:
        return f"Error running query: {str(e)}"

    return clean_text(result)

def docxfile(link, query=None):
    try:
        loader = Docx2txtLoader(link)
        data = loader.load()
    except Exception as e:
        return f"Error loading document: {str(e)}"

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(data)
    except Exception as e:
        return f"Error splitting text: {str(e)}"

    try:
        embeddings = GPT4AllEmbeddings()
        vector_store = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        return f"Error creating vector store: {str(e)}"

    try:
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
        result = qa.run(query)
    except Exception as e:
        return f"Error running query: {str(e)}"

    return clean_text(result)

def JSONLoader(link, query=None):
    try:
        loader = JSONLoader(link)
        data = loader.load()
    except Exception as e:
        return f"Error loading document: {str(e)}"

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(data)
    except Exception as e:
        return f"Error splitting text: {str(e)}"

    try:
        embeddings = GPT4AllEmbeddings()
        vector_store = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        return f"Error creating vector store: {str(e)}"

    try:
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
        result = qa.run(query)
    except Exception as e:
        return f"Error running query: {str(e)}"

    return clean_text(result)

def read_parquet(file, query):
    df = pd.read_parquet(file)
    random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=7))
    url = f"sqlite:///{random_name}.db"
    engine = create_engine(url)
    df.to_sql(f"Your", engine, index=False)
    db = SQLDatabase(engine=engine)
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent_executor = create_sql_agent(llm, db=db, memory=memory, agent_type="openai-tools", verbose=True)
    response = agent_executor.invoke({"input": query})
    return response['output']

def identify_file_and_run(uploaded_file, query):
    if uploaded_file.filename.endswith('.txt'):
        with open("temp.txt", "wb") as f:
            f.write(uploaded_file.read())
        return run_retrieval_qa_text("temp.txt", query)
    
    elif uploaded_file.filename.endswith('.pdf'):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        return run_retrieval_qa_pdf("temp.pdf", query)
    
    elif uploaded_file.filename.endswith('.csv'):
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.read())
        return run_sql_query("temp.csv", query)
    
    elif uploaded_file.filename.endswith('.parquet'):
        with open("temp.parquet", "wb") as f:
            f.write(uploaded_file.read())
        return read_parquet("temp.parquet", query)
    
    elif uploaded_file.filename.endswith('.xlsx'):
        with open("temp.xlsx", "wb") as f:
            f.write(uploaded_file.read())
        return excel("temp.xlsx", query)
    
    elif uploaded_file.filename.endswith('.docx'):
        with open("temp.docx", "wb") as f:
            f.write(uploaded_file.read())
        return docxfile("temp.docx", query)
    
    elif uploaded_file.filename.endswith('.ipynb'):
        with open("temp.ipynb", "wb") as f:
            f.write(uploaded_file.read())
        return jupyter_notebook("temp.ipynb", query)
    
    elif uploaded_file.filename.endswith('.json'):
        with open("temp.json", "wb") as f:
            f.write(uploaded_file.read())
        return JSONLoader("temp.json", query)
    
    else:
        return "Unsupported file format"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_query', methods=['POST'])
def process_query():
    start_time = time.time()  # Start timer

    url = request.form['url']
    query = request.form['query']
    task = request.form['task']

    if task == 'query_web':
        result = run_retrieval_qa_web(url, query)
    elif task == 'query_file':
        uploaded_file = request.files['file']
        result = identify_file_and_run(uploaded_file, query)
    else:
        result = "Invalid task selected."

    elapsed_time = time.time() - start_time  # Calculate elapsed time

    response = {
        'result': result.replace('\n', '<br>') if isinstance(result, str) else result,
        'time_taken': f"Time taken: {elapsed_time:.2f} seconds"
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
