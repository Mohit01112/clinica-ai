"""
Medical GP Chatbot — Streamlit + Pinecone + ChatGroq (grok-4-latest)

Bypasses langchain-pinecone VectorStore constructors (they ignore
pinecone_api_key and require PINECONE_API_KEY at import time).
Uses the official Pinecone SDK with the hardcoded key only.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from typing import List

# ---- keys first, before any Pinecone / LangChain Pinecone import ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PINECONE_API_KEY = "pcsk_Ea2mF_9cW5X6Vu9w8xTEMAEy5A8y1SvdVqqjfnPoc7GrRtoGrn3KSY7ES7ELCuejintD9"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

INDEX_NAME = "medical-chatbot"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
NAMESPACE = ""

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec

GP_SYSTEM_PROMPT = """

STRICT RESPONSE RULE
============================================================

Understand the user's question before answering.

Answer ONLY what the user is asking.

Do not automatically provide a complete analysis of the report.

The amount of information in the answer must match the question.

If the user asks for one value:
→ Give that value and, if relevant, whether it is within the
  reference range.

If the user asks whether something is normal:
→ Give a direct yes/no answer and briefly explain why.

If the user asks what something means:
→ Explain that specific result only.

If the user asks why something is abnormal:
→ Give a brief explanation of possible causes, without listing
  unnecessary information.

If the user asks about symptoms:
→ Discuss only the report findings that could reasonably relate
  to that symptom.

If the user asks for a report summary:
→ Then provide a concise overall summary.

If the user asks "What about that?", "What does that mean?",
or another ambiguous follow-up:
→ Ask which result or topic they mean.

NEVER add information simply because it exists in the report.

NEVER automatically add:
- all abnormal values
- all normal values
- tables
- symptoms
- causes
- additional tests
- treatments
- lifestyle advice
- emergency warnings
- doctor recommendations

unless they are directly relevant to the user's question.

Keep the conversation natural and concise.
You are Clinica AI, a medical report assistant.

Your job is to answer the user's question using the uploaded medical
report and relevant medical knowledge.

IMPORTANT:
Answer ONLY the question the user asked.

Do NOT summarize the entire report unless the user specifically asks
for a summary.

Do NOT provide unrelated information.

Keep answers concise and natural, like a helpful doctor explaining
something to a patient.

============================================================
ANSWER LENGTH
============================================================

Give a short, direct answer.

Usually answer in 2–5 sentences.

Only give more detail when the user's question requires it.

Do not automatically add:
- tables
- complete report summaries
- lists of all abnormal values
- causes
- symptoms
- next steps
- lifestyle advice
- urgent-care advice
- disclaimers

unless they are directly relevant to the question.

============================================================
REPORT QUESTIONS
============================================================

Use the uploaded report when answering questions about it.

Use the exact:
- test name
- result
- unit
- reference range

when available.

Never invent or change a result.

If the requested information is not available in the report,
clearly say that it cannot be verified from the report.

============================================================
EXAMPLES
============================================================

User:
"Is my hemoglobin normal?"

Good answer:
"Your hemoglobin is 12.5 g/dL, which is slightly below the
reference range of 13.0–17.0 g/dL shown in your report."

User:
"Why is my hemoglobin low?"

Good answer:
"Your hemoglobin is slightly low at 12.5 g/dL. Common causes include
iron deficiency, vitamin B12/folate deficiency, chronic disease, or
blood loss."

User:
"What is my platelet count?"

Good answer:
"Your platelet count is 150,000/cumm, which is at the lower end of
the reference range shown in your report."

User:
"Are my results normal?"

Good answer:
"Most of your reported CBC values are within the reference ranges,
but your hemoglobin is slightly low and your PCV is high."

User:
"What should I do next?"

Answer only with relevant next-step guidance based on the report.
Do not give a complete report summary.

============================================================
GENERAL MEDICAL QUESTIONS
============================================================

If the user asks a general medical question unrelated to the report,
answer the question directly using general medical knowledge.

============================================================
CONVERSATION
============================================================

Use previous conversation messages when the user refers to something
previously discussed.

For example:
User: "What about that?"
Use the previous conversation to understand what "that" refers to.

============================================================
LANGUAGE
============================================================

Respond in the same language used by the user.

============================================================
MEDICAL SAFETY
============================================================

Do not claim certainty about a diagnosis.

Do not prescribe medication or personalized doses.

If the user describes serious symptoms, recommend appropriate medical
evaluation.

============================================================
REPORT CONTEXT
============================================================

{context}
"""

def require_keys() -> None:
    if not PINECONE_API_KEY or PINECONE_API_KEY.startswith("PASTE_"):
        raise ValueError("Replace PINECONE_API_KEY at the top of app.py with your real key.")
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("PASTE_"):
        raise ValueError("Replace GROQ_API_KEY at the top of app.py with your real key.")


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_pinecone() -> Pinecone:
    require_keys()
    return Pinecone(api_key=PINECONE_API_KEY)


def get_index():
    pc = get_pinecone()
    return pc.Index(INDEX_NAME)


def ensure_index(pc: Pinecone) -> None:
    names = set()
    for idx in pc.list_indexes():
        if isinstance(idx, dict):
            names.add(idx.get("name"))
        else:
            names.add(getattr(idx, "name", None) or str(idx))
    if INDEX_NAME in names:
        return
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    for _ in range(30):
        desc = pc.describe_index(INDEX_NAME)
        status = desc.get("status", {}) if isinstance(desc, dict) else getattr(desc, "status", {})
        ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
        if ready:
            return
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for Pinecone index '{INDEX_NAME}' to be ready.")


def delete_all_vectors(index) -> None:
    try:
        index.delete(delete_all=True)
    except Exception:
        pass
    try:
        stats = index.describe_index_stats()
        namespaces = (
            stats.get("namespaces", {})
            if isinstance(stats, dict)
            else getattr(stats, "namespaces", {}) or {}
        )
        for ns in namespaces:
            try:
                index.delete(delete_all=True, namespace=ns or None)
            except Exception:
                pass
    except Exception:
        pass


def load_pdfs(uploaded_files) -> List[Document]:
    docs: List[Document] = []
    for uf in uploaded_files:
        suffix = os.path.splitext(uf.name)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()
            for p in pages:
                p.metadata = {
                    **(p.metadata or {}),
                    "source": uf.name,
                    "filename": uf.name,
                }
            docs.extend(pages)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return docs


def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def upsert_chunks(index, chunks: List[Document]) -> int:
    embeddings = get_embeddings()
    texts = [c.page_content for c in chunks]
    vectors_data = embeddings.embed_documents(texts)
    payload = []
    for i, (chunk, values) in enumerate(zip(chunks, vectors_data)):
        payload.append(
            {
                "id": f"chunk-{i}-{uuid.uuid4().hex[:8]}",
                "values": values,
                "metadata": {
                    "text": chunk.page_content[:3500],
                    "source": chunk.metadata.get("filename")
                    or chunk.metadata.get("source")
                    or "report",
                    "page": str(chunk.metadata.get("page", "")),
                },
            }
        )
    batch = 50
    ns = NAMESPACE or None
    for start in range(0, len(payload), batch):
        if ns:
            index.upsert(vectors=payload[start : start + batch], namespace=ns)
        else:
            index.upsert(vectors=payload[start : start + batch])
    return len(payload)


def ingest_reports(uploaded_files) -> int:
    raw = load_pdfs(uploaded_files)
    if not raw:
        raise ValueError("Could not extract any text from the uploaded PDF(s).")
    chunks = split_docs(raw)
    if not chunks:
        raise ValueError("PDF(s) produced no text chunks.")

    pc = get_pinecone()
    ensure_index(pc)
    index = pc.Index(INDEX_NAME)
    delete_all_vectors(index)
    return upsert_chunks(index, chunks)


def retrieve(question: str, k: int = 6) -> List[Document]:
    embeddings = get_embeddings()
    qvec = embeddings.embed_query(question)
    index = get_index()
    kwargs = {"vector": qvec, "top_k": k, "include_metadata": True}
    if NAMESPACE:
        kwargs["namespace"] = NAMESPACE
    result = index.query(**kwargs)
    matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", []) or []
    docs: List[Document] = []
    for m in matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        docs.append(
            Document(
                page_content=meta.get("text", ""),
                metadata={
                    "filename": meta.get("source", "report"),
                    "source": meta.get("source", "report"),
                    "page": meta.get("page"),
                },
            )
        )
    return docs


def format_context(docs: List[Document]) -> str:
    if not docs:
        return "(No matching excerpts were retrieved from the uploaded reports.)"
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("filename") or d.metadata.get("source") or "report"
        page = d.metadata.get("page")
        header = f"[{i}] {src}" + (f" p.{page}" if page else "")
        parts.append(f"{header}\n{d.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def build_llm() -> ChatGroq:
    require_keys()
    return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), temperature=0.2, api_key=GROQ_API_KEY)


def answer_as_gp(question: str, history: list) -> str:
    retrieved = retrieve(question)
    system = GP_SYSTEM_PROMPT.format(context=format_context(retrieved))
    messages = [SystemMessage(content=system)]
    for msg in history[-8:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))
    content = build_llm().invoke(messages).content
    if isinstance(content, list):
        bits = []
        for block in content:
            if isinstance(block, str):
                bits.append(block)
            elif isinstance(block, dict) and "text" in block:
                bits.append(block["text"])
        content = "".join(bits)
    return str(content).strip()


def init_state() -> None:
    for k, v in {
        "messages": [],
        "ready": False,
        "last_files": [],
        "chunk_count": 0,
        "error": "",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_upload() -> None:
    st.title("Clinica AI – Medical Report Assistant 🩺")
    st.write("Upload one or more test PDFs. and get answers to your questions about your results.")
    uploaded = st.file_uploader(
        "Upload medical report (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    start = st.button("Upload and start chat", type="primary", disabled=not uploaded)

    if st.session_state.error:
        st.error(st.session_state.error)

    if start:
        with st.spinner("Reading reports, resetting Pinecone, and building embeddings..."):
            try:
                n = ingest_reports(uploaded)
                st.session_state.ready = True
                st.session_state.chunk_count = n
                st.session_state.last_files = [f.name for f in uploaded]
                st.session_state.messages = []
                st.session_state.error = ""
            except Exception as e:
                st.session_state.ready = False
                st.session_state.error = f"Ingest failed: {e}"
        st.rerun()


def render_chat() -> None:
    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.title("What Would You Like to Know?")
        names = ", ".join(st.session_state.last_files) or "your reports"
        st.caption(f"Active reports: {names} · {st.session_state.chunk_count} chunks in `{INDEX_NAME}`")
    with top_r:
        if st.button("Upload a different report"):
            st.session_state.ready = False
            st.session_state.messages = []
            st.session_state.error = ""
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about your results...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Reviewing your reports..."):
            try:
                reply = answer_as_gp(prompt, st.session_state.messages[:-1])
            except Exception as e:
                reply = f"I could not complete that request: {e}"
            st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})


def main() -> None:
    st.set_page_config(page_title="GP Blood-Test Chatbot", page_icon="🩺", layout="centered")
    init_state()
    if st.session_state.ready:
        render_chat()
    else:
        render_upload()


if __name__ == "__main__":
    main()