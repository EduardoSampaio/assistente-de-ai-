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
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()  # carrega variáveis de ambiente do arquivo .env


# logica do agende de IA

def criar_chain_agente():
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
    
    def criar_contexto(prompt_usuario: str) -> str:
        return "Politica de RH da empresa XYZ. Documentos internos sobre benefícios, férias, conduta, etc."
    
    llm = ChatOpenAI(model="gemini-2.5-flash-lite", temperature=0.2, max_tokens=500)

    chain = ({"context": criar_contexto, "question": RunnablePassthrough() }  
             | prompt_template 
             | llm
             | StrOutputParser())
    return chain

# ============================================================
# INTERFACE
# ============================================================

st.title("Assistente RAG — Políticas Internas")

# carregar chain e vector store
chain = criar_chain_agente()
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