import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="Chat com PDFs", layout="wide")
st.title("Chatbot básico com PDFs")

uploaded_files = st.file_uploader(
    "Envie um ou mais PDFs",
    type="pdf",
    accept_multiple_files=True
)

def extract_text_from_pdfs(files):
    full_text = ""
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

if uploaded_files:
    with st.spinner("Extraindo texto dos PDFs..."):
        text = extract_text_from_pdfs(uploaded_files)

    if not text.strip():
        st.error("Não foi possível extrair texto dos PDFs.")
    else:
        chunks = chunk_text(text)

        st.success(f"{len(chunks)} trechos indexados.")

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks)

        embeddings = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        question = st.text_input("Faça uma pergunta sobre o conteúdo dos PDFs")

        if question:
            q_embedding = model.encode([question])
            q_embedding = np.array(q_embedding).astype("float32")

            k = 3
            distances, indices = index.search(q_embedding, k)

            st.subheader("Resposta")
            resposta = "\n\n".join([chunks[i] for i in indices[0]])
            st.write(resposta)

            st.subheader("Trechos recuperados")
            for i in indices[0]:
                st.write(chunks[i])
                st.markdown("---")
