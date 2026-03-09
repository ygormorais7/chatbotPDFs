# Chatbot básico com PDFs

## Descrição
Este projeto implementa um chatbot simples baseado no conteúdo de arquivos PDF. O usuário envia um ou mais documentos, o sistema extrai o texto, cria embeddings dos trechos e realiza busca vetorial para recuperar os trechos mais relevantes para cada pergunta.

## Tecnologias utilizadas
- Python
- Streamlit
- pypdf
- sentence-transformers
- FAISS

## Funcionalidades
- Upload de arquivos PDF
- Extração de texto
- Divisão do texto em trechos
- Geração de embeddings
- Busca vetorial
- Resposta baseada nos trechos recuperados

## Como executar
```bash
pip install -r requirements.txt
streamlit run app.py