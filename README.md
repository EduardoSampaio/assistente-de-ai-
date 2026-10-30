# Assistente RAG de Políticas Internas

Este projeto implementa um **Assistente de IA com arquitetura RAG (Retrieval-Augmented Generation)** utilizando Python, Streamlit e LangChain. O objetivo é permitir que usuários consultem documentos (como políticas de RH) de forma inteligente através de uma interface de chat.

## Funcionalidades

*   **Interface em Streamlit:** Interface web intuitiva para interação com o assistente.
*   **Arquitetura RAG:** Processamento e busca eficiente em documentos (PDFs) para fornecer respostas baseadas em contexto.
*   **Orquestração LangChain:** Utiliza o framework LangChain para gerenciar o fluxo da IA.
*   **Busca Vetorial com FAISS:** Utiliza FAISS para armazenamento e recuperação vetorial otimizada.
*   **Integração OpenAI:** Utiliza modelos da OpenAI para gerar respostas precisas baseadas nos documentos consultados.

## Requisitos

*   Python 3.x
*   Chave de API da OpenAI (configurada no arquivo `.env`)

## Instalação

1. Clone este repositório ou baixe os arquivos.
2. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv .venv
   # Ative o ambiente conforme seu sistema operacional
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure sua chave da API da OpenAI no arquivo `.env`:
   ```env
   OPENAI_API_KEY=sua_chave_aqui
   ```

## Como executar

Para iniciar a interface web, execute o comando:

```bash
streamlit run app_basic.py
```

Após executar, o Streamlit fornecerá um link local para acessar a aplicação no seu navegador.
