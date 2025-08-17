# app.py
import os
import io
import asyncio
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv


from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_bytes

from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# =======================
# CSS & Chat Templates
# =======================
CSS = '''
<style>
.chat-message {
    padding: 1rem 1.5rem; 
    border-radius: 1rem; 
    margin: 1rem 0; 
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    animation: fadeIn 0.4s ease-in-out;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: transform 0.2s ease;
}
.chat-message:hover { transform: translateY(-2px); }
.chat-message.user {
    background: linear-gradient(135deg, #2b313e, #3c4455);
    border-left: 5px solid #4ea1ff;
}
.chat-message.bot {
    background: linear-gradient(135deg, #475063, #5a6479);
    border-left: 5px solid #ffb84d;
}
.chat-message .avatar {
  width: 50px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}
.chat-message .avatar img {
  width: 48px; height: 48px;
  border-radius: 50%;
  object-fit: cover;
  background: #fff;
  padding: 8px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}
.chat-message .message {
  flex-grow: 1;
  color: #fff;
  font-size: 0.95rem;
  line-height: 1.5;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
/* sembunyikan footer default */
footer {visibility: hidden;}
</style>
'''

BOT_TMPL = '''
<div class="chat-message bot">
  <div class="avatar">
    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" alt="bot">
  </div>
  <div class="message">{{MSG}}</div>
</div>
'''

USR_TMPL = '''
<div class="chat-message user">
  <div class="avatar">
    <img src="https://cdn-icons-png.flaticon.com/512/2202/2202112.png" alt="user">
  </div>
  <div class="message">{{MSG}}</div>
</div>
'''


# =======================
# AsyncIO setup (untuk grpc aio di Google client)
# =======================
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
nest_asyncio.apply()


# =======================
# PDF Utilities
# =======================
def read_pdfs_to_bytes(uploaded_files):
    """
    Kembalikan list of (name, bytes) agar bisa dipakai ulang untuk
    text-extraction & OCR tanpa masalah pointer/seek.
    """
    files = []
    for f in uploaded_files:
        data = f.read()
        files.append((f.name, data))
    return files


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text menggunakan PyPDF2 dari bytes."""
    text = ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    except Exception:
        # Kalau gagal parse, return kosong; OCR akan jalan sebagai fallback
        return ""
    return text


def ocr_text_from_pdf_bytes(pdf_bytes: bytes, langs: str = "eng") -> str:
    """OCR setiap halaman PDF (render ke image dulu)."""
    text = ""
    try:
        images = convert_from_bytes(pdf_bytes)  # requires poppler on Linux; on mac via brew it's fine
        for img in images:
            text += pytesseract.image_to_string(img, lang=langs)
    except Exception:
        return ""
    return text


def get_all_text(files_bytes):
    """
    Gabungkan teks dari semua file.
    - Coba text-based extraction dulu
    - Jika hasil kosong, fallback ke OCR
    """
    combined = ""
    used_ocr = False

    for name, data in files_bytes:
        t = extract_text_from_pdf_bytes(data)
        if not t.strip():
            # fallback OCR
            ocr_t = ocr_text_from_pdf_bytes(data, langs="eng")  # tambahkan "ind" jika perlu: "eng+ind"
            if ocr_t.strip():
                used_ocr = True
                combined += ocr_t + "\n"
        else:
            combined += t + "\n"

    return combined.strip(), used_ocr


# =======================
# RAG Building Blocks
# =======================
def split_text(text: str):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)


def build_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)


def build_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


# =======================
# Chat UI Helpers
# =======================
def render_user(msg: str):
    st.write(USR_TMPL.replace("{{MSG}}", msg), unsafe_allow_html=True)

def render_bot(msg: str):
    st.write(BOT_TMPL.replace("{{MSG}}", msg), unsafe_allow_html=True)

def ask_question(query: str):
    """Kirim pertanyaan ke chain, render ke UI."""
    if not st.session_state.get("conversation"):
        st.warning("Proses dulu PDF kamu sebelum bertanya.")
        return
    response = st.session_state.conversation({"question": query})
    st.session_state.chat_history = response.get("chat_history", [])
    render_user(query)
    # ambil jawaban terakhir dari bot
    if st.session_state.chat_history:
        bot_msg = st.session_state.chat_history[-1].content
        render_bot(bot_msg)


# =======================
# MAIN APP
# =======================
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📄")

    # Render CSS
    st.markdown(CSS, unsafe_allow_html=True)

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ Tentang Chatbot")
        st.markdown(
            """
**Model AI yang digunakan**  
👉 **Google Gemini** *(aktif untuk app ini)*

**Peran AI dalam chatbot**  
👉 AI memahami pertanyaan pengguna, menganalisis isi PDF, lalu memberikan jawaban yang relevan.

**Langkah pakai:**
1) Upload 1–n PDF  
2) Klik **Process**  
3) Pilih salah satu pertanyaan rekomendasi atau ketik pertanyaanmu
            """
        )
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: grey;'>Made by @helmyyoga</p>", unsafe_allow_html=True)

    st.title("📄 Chat dengan PDF")
    st.caption("Unggah PDF kamu, lalu tanya apa saja tentang isinya.")

    # Init session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Upload & process
    uploaded = st.file_uploader("Upload satu atau beberapa PDF", type="pdf", accept_multiple_files=True)
    if st.button("Process"):
        if not uploaded:
            st.error("Silakan upload minimal 1 PDF dulu.")
        else:
            with st.spinner("Processing PDF..."):
                files_bytes = read_pdfs_to_bytes(uploaded)
                full_text, used_ocr = get_all_text(files_bytes)

                if not full_text.strip():
                    st.error("❌ Tidak ada teks yang bisa diekstrak dari PDF (bahkan dengan OCR).")
                    return

                chunks = split_text(full_text)
                if not chunks:
                    st.error("❌ Tidak ada potongan teks yang valid setelah proses splitting.")
                    return

                vectorstore = build_vectorstore(chunks)
                st.session_state.conversation = build_conversation_chain(vectorstore)

            st.success("✅ PDF berhasil diproses!" + (" (menggunakan OCR)" if used_ocr else ""))

            # Bot auto prompt + quick suggestions
            intro = (
                "Halo 👋, dokumenmu sudah siap dipelajari!\n\n"
                "Apa yang ingin kamu ketahui dari dokumen ini?\n\n"
                "💡 Contoh pertanyaan:\n"
                "• Ringkas isi dokumen ini\n"
                "• Apa saja poin-poin penting?\n"
                "• Sebutkan angka/tanggal/nama penting\n"
                "• Adakah kesimpulan atau rekomendasi?\n"
                "• Jawab pertanyaan spesifik tentang topik X\n"
            )
            render_bot(intro)

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                if st.button("Ringkas dokumen"):
                    ask_question("Ringkas isi dokumen ini secara singkat dan terstruktur.")
            with col2:
                if st.button("Poin penting"):
                    ask_question("Apa saja poin-poin penting dari dokumen ini? Buat bullet points.")
            with col3:
                if st.button("Angka & tanggal"):
                    ask_question("Sebutkan angka, tanggal, atau metrik penting beserta konteksnya.")
            with col4:
                if st.button("Nama & entitas"):
                    ask_question("Sebutkan nama orang, organisasi, lokasi, atau entitas penting yang disebutkan.")
            with col5:
                if st.button("Kesimpulan"):
                    ask_question("Apa kesimpulan utama dokumen ini? Sertakan rekomendasi bila ada.")

    # Free-form Q&A setelah sudah ada conversation
    if st.session_state.conversation:
        user_q = st.text_input("❓ Pertanyaanmu tentang PDF")
        if user_q:
            ask_question(user_q)


if __name__ == "__main__":
    main()
