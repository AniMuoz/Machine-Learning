import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# ---- Cargar dataset de smartphones ----
@st.cache_data
def cargar_dataset():
    return pd.read_csv("Smartphone_data_2025.csv", encoding="latin1")

df = cargar_dataset()

# Mostrar parte del dataset en el sidebar
st.sidebar.title("📱 Dataset de Smartphones")
st.sidebar.dataframe(df.head(10))

# Convertir los datos a texto para pasárselos al modelo
def obtener_contexto(df, max_filas=100):
    pd.set_option('display.max_rows', max_filas)
    return df.head(max_filas).to_string(index=False)


contexto_dataset = obtener_contexto(df)

# ---- Filtro por palabras clave ----
def es_pregunta_sobre_smartphones(texto):
    keywords = ["smartphone", "celular", "móvil", "telefono", "marca", "modelo", "iphone", "samsung",
                "xiaomi", "batería", "pantalla", "procesador", "android", "apple"]
    texto_lower = texto.lower()
    return any(k in texto_lower for k in keywords)

# ---- Construir prompt con instrucciones ----
def construir_prompt_usuario(pregunta_usuario):
    instrucciones = (
        "Eres un asistente experto en smartphones. Solo debes responder preguntas relacionadas con smartphones, "
        "incluyendo modelos, precios, especificaciones técnicas y comparaciones. "
        "Usa la siguiente información sobre smartphones para responder preguntas y hacer comparaciones, los precios los presentaras en dolares estadounidences (USD)"
        "Si la pregunta no es sobre smartphones, responde con: "
        "'Lo siento, solo puedo responder preguntas relacionadas con smartphones.'\n\n"
        )
    return instrucciones + f"Información disponible:\n{contexto_dataset}\n\nPregunta: {pregunta_usuario}"

# Inicializar modelo Ollama
llm = Ollama(model="mistral:latest", base_url="http://localhost:11434", verbose=True)
#llm = Ollama(model="deepseek-r1:7b", base_url="http://localhost:11434", verbose=True)

# Función para enviar el prompt
def sendPrompt(prompt):
    contexto = (
        "Eres un experto en tecnología móvil que respondera siempre en español y que recibe preguntas en español, no responderas otra pregunta que no sea sobre smartphones. Usa la siguiente información sobre smartphones para responder preguntas y hacer comparaciones, los precios los presentaras en dolares estadounidences (USD):\n\n"
        f"{contexto_dataset}\n\n"
        f"Pregunta del usuario: {prompt}"
    )
    response = llm.invoke(contexto)
    return response

# Interfaz con Streamlit
st.title("📱 Comparador de Smartphones del 2024 y 2025 con IA")

# Mensajes en sesión
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Pregúntame sobre modelos de smartphones y los comparo por ti."}
    ]

# Entrada del usuario
if prompt := st.chat_input("Escribe tu pregunta sobre smartphones..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Si el último mensaje es del usuario, generar respuesta
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Analizando smartphones..."):
            response = sendPrompt(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
