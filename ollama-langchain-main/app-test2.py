import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
import os

# ---- CONFIGURACIÓN ----
CSV_PATH = "Smartphone_data_2025.csv"
DB_PATH = "vector_db_smartphones"

# ---- FUNCIONES AUXILIARES ----
def crear_documentos(df):
    documentos = []
    for _, fila in df.iterrows():
        contenido = "\n".join([f"{col}: {fila[col]}" for col in df.columns])
        documentos.append(Document(page_content=contenido))
    return documentos

def construir_qa_chain():
    embeddings = OllamaEmbeddings(model="mistral", base_url="http://localhost:11434")

    if not os.path.exists(DB_PATH):
        df = pd.read_csv(CSV_PATH, encoding="latin1")
        documentos = crear_documentos(df)
        vectorstore = FAISS.from_documents(documentos, embeddings)
        vectorstore.save_local(DB_PATH)
    else:
        vectorstore = FAISS.load_local(DB_PATH, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = Ollama(model="mistral", base_url="http://localhost:11434")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

def es_pregunta_sobre_smartphones(texto):
    keywords = ["smartphone", "celular", "móvil", "modelo", "iphone", "samsung",
                "xiaomi", "batería", "pantalla", "procesador", "android", "apple"]
    texto_lower = texto.lower()
    return any(k in texto_lower for k in keywords)

# ---- INICIALIZAR APP ----
st.set_page_config(page_title="Asistente de Smartphones", page_icon="📱")
st.title("📱 Asistente IA de Smartphones")

# Cargar la cadena QA con FAISS + embeddings
qa_chain = construir_qa_chain()

# Historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy un experto en smartphones. ¿Qué querés saber o comparar?"}
    ]

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Entrada del usuario
if prompt := st.chat_input("Escribí tu consulta sobre smartphones..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            if not es_pregunta_sobre_smartphones(prompt):
                respuesta = "Lo siento, solo puedo responder preguntas relacionadas con smartphones."
            else:
                respuesta = qa_chain.run(prompt)

            st.write(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

