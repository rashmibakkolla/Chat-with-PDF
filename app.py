

# groq code 
import streamlit as st
import os
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

from collections import Counter


# ---------------- LOAD ENV ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please add it to .env file")
    st.stop()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Chat with PDF (Free RAG)", layout="wide")
st.header("📄 Chat with PDF using Free RAG (Groq + Open Models)")

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = []

# ---------------- UPLOAD PDF ----------------
pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    reader = PdfReader(pdf)
    raw_text = ""
    page_texts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
            page_texts.append(text)

    # ---------------- SUMMARY ----------------
    with st.expander("📑 PDF Summary"):
        summary_llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.2
        )

        prompt = f"Summarize the following document in bullet points:\n{raw_text[:3000]}"
        summary = summary_llm.invoke(prompt)
        st.write(summary.content)

    # ---------------- TEXT ANALYSIS ----------------
    # with st.expander("📊 Text Analysis"):
    #     words = raw_text.split()
    #     freq = Counter(words)

    #     st.write("**Top 10 Frequent Words**")
    #     st.table(freq.most_common(10))

    #     wc = WordCloud(
    #         width=800,
    #         height=400,
    #         background_color="white"
    #     ).generate(raw_text)

    #     fig, ax = plt.subplots(figsize=(10, 5))
    #     ax.imshow(wc, interpolation="bilinear")
    #     ax.axis("off")
    #     st.pyplot(fig)

    # ---------------- SPLITTING ----------------
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(raw_text)

    # ---------------- EMBEDDINGS (FREE + OFFLINE) ----------------
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # ---------------- QA MODEL ----------------
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

    chain = load_qa_chain(llm, chain_type="stuff")

    # ---------------- PAGE VIEW ----------------
    with st.expander("📄 View PDF by Page"):
        page_no = st.slider("Select Page", 1, len(page_texts), 1)
        st.text_area(
            "Page Content",
            page_texts[page_no - 1],
            height=250
        )

    # ---------------- QUESTIONS ----------------
    st.subheader("❓ Ask Questions (one per line)")
    questions = st.text_area("Enter your questions")

    if questions:
        for q in questions.split("\n"):
            q = q.strip()
            if not q:
                continue

            docs = vectorstore.similarity_search(q)
            response = chain(
                {"input_documents": docs, "question": q},
                return_only_outputs=True
            )

            answer = response["output_text"]

            st.markdown(f"**Q:** {q}")
            st.write(answer)

            if st.button("⭐ Bookmark", key=q):
                st.session_state.bookmarks.append((q, answer))

            st.session_state.chat_history.append((q, answer))

    # ---------------- DOWNLOAD CHAT ----------------
    if st.session_state.chat_history:
        chat_text = "\n\n".join(
            [f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history]
        )
        st.download_button(
            "📥 Download Chat History",
            chat_text,
            "chat_history.txt"
        )

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⭐ Bookmarks")
    for q, a in st.session_state.bookmarks[::-1]:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")

    st.title("🕘 Chat History")
    for q, a in st.session_state.chat_history[::-1]:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")

    if st.button("🗑️ Clear All"):
        st.session_state.chat_history = []
        st.session_state.bookmarks = []
