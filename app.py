# app.py — ITSM RAG Assistant (FULL FEATURED)
# Features: Sources, Sidebar, History, PDF Download, Voice Input, Chat Bubbles,
# Dark Mode toggle, Model selector (Ollama), Export Chat History (TXT/JSON)
# Compatible with LangChain 0.3+

import streamlit as st
import json
from io import BytesIO
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# optional features (voice)
try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

# PDF
try:
    from reportlab.pdfgen import canvas
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# -------- CONFIG ----------
VECTOR_DB_DIR = "vectordb"

# Available models - update these to models you have pulled with `ollama pull`
MODEL_OPTIONS = [
    "llama3",
    "llama3:instruct",
    "phi3",
    # add other models you pulled locally
]

# ---------------------------
# Utilities: PDF generation
# ---------------------------
def generate_pdf(answer_text: str) -> BytesIO:
    buffer = BytesIO()
    if not PDF_AVAILABLE:
        buffer.write(answer_text.encode("utf-8"))
        buffer.seek(0)
        return buffer

    pdf = canvas.Canvas(buffer)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(40, 800, "ITSM RAG Assistant - Answer")
    y = 780
    for line in answer_text.split("\n"):
        pdf.drawString(40, y, line)
        y -= 14
        if y < 60:
            pdf.showPage()
            y = 800
    pdf.save()
    buffer.seek(0)
    return buffer

# ---------------------------
# Utilities: voice -> text
# ---------------------------
def voice_to_text(timeout=5, phrase_time_limit=8):
    if not VOICE_AVAILABLE:
        return ""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... speak now")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        text = recognizer.recognize_google(audio)
        st.success(f"Recognized: {text}")
        return text
    except Exception as e:
        st.error(f"Voice input failed: {e}")
        return ""

# ---------------------------
# Vector DB loader
# ---------------------------
@st.cache_resource
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )
    return vectordb

# ---------------------------
# RAG chain builder
# ---------------------------
def build_rag_chain_and_retriever(selected_model: str, k: int = 3):
    vectordb = load_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    llm = Ollama(model=selected_model)

    prompt = ChatPromptTemplate.from_template(
        """
        You are an ITSM expert. Use ONLY the information in the context.
        If the answer is not found in the documents, reply:
        "Not found in provided ITSM documents."

        Context:
        {context}

        Question:
        {question}
        """
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain, retriever

# ---------------------------
# UI / Main app
# ---------------------------
def main():
    st.set_page_config(page_title="ITSM RAG Assistant", layout="wide")

    # ------- Top bar: theme and model selection -------
    cols = st.columns([3, 2, 2])
    with cols[0]:
        st.title("🧠 ITSM RAG Assistant")
    with cols[1]:
        # Model selector
        selected_model = st.selectbox("Model", MODEL_OPTIONS, index=0, help="Choose a local Ollama model. Pull it with `ollama pull <model>`.")
    with cols[2]:
        dark_mode = st.checkbox("🌙 Dark mode", value=False)

    # Inject minimal CSS for dark mode & chat bubbles
    if dark_mode:
        st.markdown(
            """
            <style>
            body { background-color: #0b1020; color: #e6eef8; }
            .stApp .block-container { background-color: #0b1020; }
            .user-bubble { background-color:#2b7a2b;color:white;padding:10px;border-radius:12px;margin:8px 0;width:60%; }
            .assistant-bubble { background-color:#2b2b2b;color:white;padding:10px;border-radius:12px;margin:8px 0;width:60%; }
            </style>
            """, unsafe_allow_html=True
        )
        user_bubble_style = "background-color:#2b7a2b;color:white"
        assistant_bubble_style = "background-color:#2b2b2b;color:white"
    else:
        st.markdown(
            """
            <style>
            .user-bubble { background-color:#DCF8C6;color:black;padding:10px;border-radius:12px;margin:8px 0;width:60%; }
            .assistant-bubble { background-color:#EAEAEA;color:black;padding:10px;border-radius:12px;margin:8px 0;width:60%; }
            </style>
            """, unsafe_allow_html=True
        )
        user_bubble_style = "background-color:#DCF8C6;color:black"
        assistant_bubble_style = "background-color:#EAEAEA;color:black"

    # Sidebar with instructions
    with st.sidebar:
        st.header("📘 How to use")
        st.markdown(
            """
            - Choose a **local** model (pull with `ollama pull <model>`).
            - Ask ITSM questions (Incident, Major Incident, CMDB, SLA, etc.)
            - Use **Voice** (if supported) or type your question.
            - Download answers as PDF, or export chat history.
            """
        )
        st.markdown("### Example Questions")
        st.markdown("- What is the Incident Management process?")
        st.markdown("- Explain Major Incident workflow.")
        st.markdown("- Show CMDB structure.")
        st.markdown("---")
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Build RAG chain for the selected model (not cached by model by default; rebuild each selection)
    rag_chain, retriever = build_rag_chain_and_retriever(selected_model, k=3)

    # Input area (voice + text)
    input_col, action_col = st.columns([5,1])
    with input_col:
        user_input = st.text_input("Your question:", key="user_input")
    with action_col:
        if VOICE_AVAILABLE and st.button("🎤 Voice"):
            recognized = voice_to_text()
            if recognized:
                st.session_state.user_input = recognized
                user_input = recognized

    # Submit handling
    if st.button("Ask"):
        if user_input and user_input.strip():
            with st.spinner("Thinking..."):
                answer = rag_chain.invoke(user_input)
                docs = retriever.invoke(user_input)

            # Append to history with metadata
            timestamp = datetime.utcnow().isoformat()
            st.session_state.chat_history.append({
                "timestamp": timestamp,
                "role": "user",
                "text": user_input
            })
            st.session_state.chat_history.append({
                "timestamp": timestamp,
                "role": "assistant",
                "text": answer if isinstance(answer, str) else str(answer)
            })

            # Display answer and sources
            st.markdown("### ✅ Answer")
            st.write(answer)

            # Download answer as PDF
            pdf_buffer = generate_pdf(str(answer))
            st.download_button(
                label="📄 Download Answer as PDF",
                data=pdf_buffer,
                file_name=f"itsm_answer_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )

            # Sources
            st.markdown("### 📚 Sources Used")
            if not docs:
                st.write("No sources found.")
            else:
                for i, doc in enumerate(docs, start=1):
                    st.markdown(f"**Source {i}:** {doc.metadata.get('source','Unknown')}")
                    st.write(doc.page_content[:400] + "...")
                    st.markdown("---")

    # Chat history displayed as bubbles with export buttons
    st.markdown("## 💬 Chat History")
    for item in st.session_state.chat_history:
        if item["role"] == "user":
            st.markdown(f"<div class='user-bubble'><b>You:</b> {item['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-bubble'><b>Assistant:</b> {item['text']}</div>", unsafe_allow_html=True)

    # Export chat history section
    st.markdown("---")
    st.markdown("### ⤓ Export Chat History")
    if st.session_state.chat_history:
        # TXT export
        txt_lines = []
        for entry in st.session_state.chat_history:
            dt = entry.get("timestamp", "")
            role = entry.get("role", "")
            text = entry.get("text", "")
            txt_lines.append(f"[{dt}] {role.upper()}: {text}")
        txt_data = "\n".join(txt_lines).encode("utf-8")
        st.download_button("Download as TXT", data=txt_data, file_name="itsm_chat_history.txt", mime="text/plain")

        # JSON export
        json_data = json.dumps(st.session_state.chat_history, indent=2).encode("utf-8")
        st.download_button("Download as JSON", data=json_data, file_name="itsm_chat_history.json", mime="application/json")
    else:
        st.write("No chat history to export.")

if __name__ == "__main__":
    main()
