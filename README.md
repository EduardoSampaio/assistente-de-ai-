# Assistente RAG de Políticas Internas

Este projeto implementa um **Assistente de IA com arquitetura RAG (Retrieval-Augmented Generation)**. Ele permite que os usuários realizem consultas inteligentes em documentos (como políticas de RH) através de uma interface de chat, garantindo respostas baseadas no contexto dos próprios arquivos fornecidos.

## Tecnologias e Soluções Utilizadas

*   **Linguagem:** Python.
*   **Interface:** [Streamlit](https://streamlit.io/) para a criação de uma interface web interativa e amigável.
*   **Orquestração de IA:** [LangChain](https://www.langchain.com/) para gerenciar o fluxo de dados, prompts e a orquestração da cadeia RAG.
*   **Processamento de Linguagem:** Modelos do **Grok (xAI)**, orquestrados via **Google Cloud Vertex AI** para gerar respostas precisas e contextuais.
*   **Recuperação Vetorial:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) para armazenamento e busca eficiente de similaridade semântica nos documentos.
*   **Embeddings:** Modelos do HuggingFace (via `sentence-transformers`) para converter o texto dos documentos em vetores numéricos.
*   **Carregamento de Dados:** `PyPDFLoader` para processar arquivos PDF.

## Funcionalidades

*   **Interface em Streamlit:** Interface web intuitiva para interação com o assistente.
*   **Arquitetura RAG:** Processamento e busca eficiente em documentos (PDFs) para fornecer respostas baseadas em contexto real.
*   **Orquestração LangChain:** Gerenciamento robusto do fluxo da IA, desde a recuperação do contexto até a geração da resposta.
*   **Busca Vetorial com FAISS:** Busca ultra-rápida de documentos relevantes.
*   **Integração com Modelos Grok:**.

## Requisitos

*   Python 3.x
*   Chave de API da GrokCloud (configurada no arquivo `.env`)

## Instalação

1. Clone este repositório ou baixe os arquivos.
2. Crie um ambiente virtual (recomendado):
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
   GROQ_API_KEY=sua_chave_aqui
   ```

## Como executar

Para iniciar a interface web, execute o comando:

```bash
streamlit run app_basic.py
```

Após executar, o Streamlit fornecerá um link local para acessar a aplicação no seu navegador.
