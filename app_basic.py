"""
Assistente RAG de Políticas Internas (RH)
- Streamlit (interface)
- LangChain com LCEL (orquestração)
- FAISS (banco vetorial)
- OpenAI (embeddings + LLM)
"""

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()  # carrega variáveis de ambiente do arquivo .env

def carregar_documentos():
    """
    Carrega o arquivo PDF 'politica_rh.pdf', divide o conteúdo em chunks 
    (blocos de texto) menores para otimizar a busca e o processamento de RAG.
    
    Returns:
        list: Lista de documentos processados (chunks).
    """
    # Carregar documentos PDF
    loader = PyPDFLoader("politica_rh.pdf")
    documentos = loader.load()

    # Dividir documentos em chunks menores
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documentos)

    return chunks

# Processo de embeddings e criação do banco vetorial
def criar_vectorstore():
    """
    Cria a base de dados vetorial (FAISS) a partir dos chunks de texto, 
    utilizando embeddings do HuggingFace para representar o significado do texto.
    
    Returns:
        FAISS: O objeto de banco vetorial populado.
    """
    chunks = carregar_documentos()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

banco_vetores = criar_vectorstore()


# logica do agende de IA
def criar_chain_agente(vectorstore):
    """
    Cria a cadeia de processamento (chain) do assistente de IA.
    Configura o prompt, o retriever (buscador no banco vetorial) e o modelo LLM.
    
    Args:
        vectorstore (FAISS): O banco vetorial usado para busca de contexto.
        
    Returns:
        chain: A cadeia configurada com LangChain (LCEL) pronta para uso.
    """
    prompt_template = ChatPromptTemplate.from_template(
        """
    Você é um assistente de RH que responde perguntas sobre políticas internas da empresa.
    Use APENAS as informações do contexto para responder. Se não souber a resposta, diga que não sabe.
    Responda de forma clara e objetiva, sem inventar informações.
    Responda em português.
    Responda as perguntas com base no
    contexto: {context} fornecido. 
    A pergunta: {question}
    A Resposta: 
    """)
    
  
    retriever = banco_vetores.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    chain = ({"context": retriever, "question": RunnablePassthrough() }  
             | prompt_template 
             | llm
             | StrOutputParser())
    return chain


# ============================================================
# INTERFACE
# ============================================================

st.title("Assistente RAG — Políticas Internas")

# carregar chain e vector store
chain = criar_chain_agente(banco_vetores)
# inicia a lista de mensagens
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# exibe as mensagens na tela
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Pergunte sobre as políticas da empresa...")
if prompt :
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando..."):
            answer = chain.invoke(prompt)
            st.write(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})