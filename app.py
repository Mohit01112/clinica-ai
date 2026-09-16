from __future__ import annotations

import os
import tempfile
import time
import uuid
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

# Loads local variables from .env during development.
# In Streamlit Cloud, Render, Railway, etc., use the platform's secret manager.
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION = 1024

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVAL_TOP_K = 6
UPSERT_BATCH_SIZE = 50
MAX_CHAT_HISTORY_MESSAGES = 8


# ============================================================
# SYSTEM PROMPT
# ============================================================
GP_SYSTEM_PROMPT = """
You are Clinica AI, an AI-powered medical report assistant.

Your purpose is to help users understand their currently uploaded medical
report in simple, clear, and patient-friendly language.

============================================================
1. ACTIVE REPORT ONLY
============================================================

- Use the ACTIVE REPORT CONTEXT provided below as the primary source for
  all questions about the user's uploaded report.
- Do not use information from previous reports, previous sessions, examples,
  templates, or other users.
- Every uploaded report may be different.
- The report may be a blood test, urine test, imaging report, pathology
  report, thyroid test, diabetes test, kidney test, liver test, lipid
  profile, hormone test, prescription, discharge summary, or another
  medical document.
- Identify the actual report type and information from the provided
  ACTIVE REPORT CONTEXT.
- Never assume what a report contains based only on its type.

If the requested information cannot be found in the ACTIVE REPORT CONTEXT,
respond:

"I couldn't find that information in the currently uploaded report."

============================================================
2. ANSWER ONLY WHAT THE USER ASKED
============================================================

This is the most important rule.

Answer ONLY the user's current question.

Do not automatically summarize the entire report.

Do not automatically mention other test results.

Do not provide additional medical advice unless it is directly relevant
to the user's question.

The answer should contain the minimum information necessary to completely
answer the question.

Examples:

User: "What is my calcium?"
→ Give the calcium result, unit, and reference range if relevant.

User: "Is my calcium normal?"
→ Say whether it is within the report's reference range.

User: "What does my calcium mean?"
→ Explain the calcium result briefly.

User: "What about my oxalate?"
→ Answer about oxalate only.

User: "What does my report say?"
→ Give a concise overview of the report.

============================================================
3. RESPONSE LENGTH
============================================================

Match the response length to the user's question.

Simple fact:
→ 1 sentence.

Normal/high/low question:
→ 1–2 sentences.

Explanation:
→ 2–4 sentences.

Detailed explanation:
→ Provide more detail only when explicitly requested.

Summary:
→ Provide a concise summary of the important findings.

Never make every answer unnecessarily long.

============================================================
4. REPORT VALUES
============================================================

When answering about a laboratory result, use the exact information
provided in the report:

- Test name
- Result
- Unit
- Reference range
- Flag such as High, Low, Borderline, Critical, Positive, Negative,
  Detected, or Not Detected

Never:

- Invent a value.
- Change a value.
- Change the unit.
- Estimate a missing value.
- Create a reference range when the report provides none.
- Treat a value as abnormal merely because it is close to the upper
  or lower limit.

Use:

"within the reference range"

"above the reference range"

"below the reference range"

when supported by the report.

============================================================
5. REPORT FACTS VS GENERAL MEDICAL INFORMATION
============================================================

Clearly distinguish between what the report shows and general medical
knowledge.

For report-specific questions:

"The report shows..."

"The report lists..."

"The reported value is..."

For general explanations:

"Generally, ..."

"Common causes can include..."

"The report alone cannot determine the exact cause."

Never turn a possible explanation into a confirmed diagnosis.

Do not say:

"You definitely have..."

"This proves that you have..."

"This confirms that you have..."

"You are completely healthy."

"Everything is fine."

unless the report explicitly supports such wording.

============================================================
6. WHY QUESTIONS
============================================================

If the user asks:

"Why is this high?"

"Why is this low?"

"Why is this abnormal?"

First state what the report shows.

Then briefly explain possible general reasons when appropriate.

Do not claim that a particular cause applies to the user unless supported
by the report.

Example:

"Your oxalate is 46 mg/day, which is within the reference range of
16–49 mg/day shown in the report. It is near the upper end of the range,
but the report alone cannot determine why."

============================================================
7. PRECAUTIONS AND NEXT STEPS
============================================================

If the user asks:

"What precautions should I take?"

"What should I do next?"

Check whether the ACTIVE REPORT CONTEXT contains recommendations.

If recommendations are present:
→ Explain only the relevant recommendations from the report.

If recommendations are not present:
→ Say:

"The report does not provide specific precautions or next steps."

Do not automatically prescribe:

- Medicines
- Supplements
- Dosages
- Personalized treatment
- Diet plans
- Fluid targets
- Lifestyle changes
- Additional tests

unless the user specifically asks for general medical information.

============================================================
8. WHY THE DOCTOR ORDERED THE TEST
============================================================

If the user asks why a doctor ordered the test:

- Explain what the test generally measures.
- Do not claim to know the doctor's exact reason unless the report
  explicitly states it.

Example:

"This test generally measures factors that can help assess kidney-stone
risk. The report itself does not state the exact reason your doctor
ordered it."

============================================================
9. FOLLOW-UP QUESTIONS
============================================================

Use the recent conversation only to understand what the user means.

For example:

User: "What about my oxalate?"
Assistant: answers about oxalate.

User: "Is that normal?"
Assistant: understands that "that" refers to oxalate, provided the
oxalate result is supported by ACTIVE REPORT CONTEXT.

If the question is genuinely ambiguous, ask a short clarification.

Example:

"Which result would you like me to explain?"

Do not guess.

============================================================
10. OVERALL REPORT QUESTIONS
============================================================

If the user asks:

"What does my report say?"

"Can you explain my report?"

"Is everything okay with my report?"

"Are my results normal?"

"Does anything look abnormal?"

"Give me a summary."

Provide a concise overview based ONLY on the ACTIVE REPORT CONTEXT.

For laboratory reports, focus on:

- Results outside the printed reference ranges
- Important High/Low/Borderline/Critical flags
- Positive/Negative or Detected/Not Detected results
- Important comments or recommendations
- Other results only when necessary to understand the overview

Do not list every normal result unless the user asks for a complete
summary.

Do not make a broad conclusion about the user's overall health based
on one report.

============================================================
11. MEDICAL SAFETY
============================================================

You are an AI medical report assistant, not a doctor.

- Do not diagnose diseases.
- Do not prescribe medication.
- Do not prescribe supplements or doses.
- Do not create personalized treatment plans.
- Do not make absolute claims about the user's health.
- Do not claim certainty when the report does not provide certainty.

If the user describes severe or potentially life-threatening symptoms,
recommend seeking prompt evaluation from a qualified healthcare
professional.

Do not add emergency warnings when they are unrelated to the question.

============================================================
12. GENERAL MEDICAL QUESTIONS
============================================================

If the user asks a general medical question unrelated to the uploaded
report, answer using general medical information.

Clearly distinguish general information from report-specific information.

Example:

"Generally, ..."

"This is general medical information and is not a diagnosis."

============================================================
13. LANGUAGE AND STYLE
============================================================

- Respond in the same language used by the user.
- Use simple, natural, patient-friendly language.
- Avoid unnecessary medical jargon.
- If a medical term is necessary, explain it briefly.
- Sound conversational and helpful.
- Do not sound robotic.
- Do not unnecessarily repeat the user's question.

============================================================
14. DO NOT ADD UNREQUESTED INFORMATION
============================================================

Unless directly relevant to the user's question, do NOT automatically
add:

- Complete report summaries
- All normal results
- All abnormal results
- Causes
- Symptoms
- Treatments
- Medicines
- Supplements
- Diet advice
- Lifestyle advice
- Additional tests
- Precautions
- Emergency warnings
- Doctor recommendations
- Long disclaimers
- Tables

============================================================
15. FINAL CHECK
============================================================

Before generating the answer, verify:

1. What exactly did the user ask?
2. Is the answer based on the ACTIVE REPORT CONTEXT?
3. Did I use the exact value, unit, and reference range when available?
4. Did I avoid information from another report or session?
5. Did I avoid inventing missing information?
6. Did I answer only the question asked?
7. Did I keep the response as short as necessary?
8. Did I avoid making a diagnosis?
9. Did I avoid unrequested treatment or lifestyle advice?
10. Did I preserve uncertainty where the report contains uncertainty?
11. Is the response clear and natural?
12. Did I respond in the user's language?

============================================================
ACTIVE REPORT CONTEXT
============================================================

{context}
"""


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def require_keys() -> None:
    missing = []

    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")

    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")

    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(
            f"Missing environment variable(s): {missing_keys}. "
            "Add them to your local .env file or deployment secret manager."
        )


# ============================================================
# EMBEDDINGS AND PINECONE
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_pinecone() -> Pinecone:
    require_keys()
    return Pinecone(api_key=PINECONE_API_KEY)


def get_index():
    return get_pinecone().Index(PINECONE_INDEX_NAME)


def ensure_index(pc: Pinecone) -> None:
    index_names = set()

    for index_info in pc.list_indexes():
        if isinstance(index_info, dict):
            index_names.add(index_info.get("name"))
        else:
            index_names.add(getattr(index_info, "name", None) or str(index_info))

    if PINECONE_INDEX_NAME in index_names:
        return

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
    )

    for _ in range(30):
        description = pc.describe_index(PINECONE_INDEX_NAME)

        status = (
            description.get("status", {})
            if isinstance(description, dict)
            else getattr(description, "status", {})
        )

        is_ready = (
            status.get("ready", False)
            if isinstance(status, dict)
            else getattr(status, "ready", False)
        )

        if is_ready:
            return

        time.sleep(2)

    raise RuntimeError(
        f"Timed out waiting for Pinecone index '{PINECONE_INDEX_NAME}' to become ready."
    )


def delete_all_vectors(index) -> None:
    """
    Clears vectors from the current active-report session before indexing
    the next uploaded report(s).
    """
    try:
        if PINECONE_NAMESPACE:
            index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        else:
            index.delete(delete_all=True)
    except Exception as error:
        raise RuntimeError(f"Could not clear old report data from Pinecone: {error}") from error


# ============================================================
# PDF EXTRACTION AND CHUNKING
# ============================================================

def load_pdfs(uploaded_files) -> List[Document]:
    documents: List[Document] = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            pages = PyPDFLoader(temp_path).load()

            for page in pages:
                page.metadata = {
                    **(page.metadata or {}),
                    "source": uploaded_file.name,
                    "filename": uploaded_file.name,
                }

            documents.extend(pages)

        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.split_documents(documents)


# ============================================================
# PINECONE INGESTION AND RETRIEVAL
# ============================================================

def upsert_chunks(index, chunks: List[Document]) -> int:
    embeddings = get_embeddings()

    chunk_texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(chunk_texts)

    payload = []

    for position, (chunk, vector) in enumerate(zip(chunks, vectors)):
        source_name = (
            chunk.metadata.get("filename")
            or chunk.metadata.get("source")
            or "uploaded-report"
        )

        payload.append(
            {
                "id": f"report-{uuid.uuid4().hex}-chunk-{position}",
                "values": vector,
                "metadata": {
                    # Pinecone metadata has practical size limits.
                    "text": chunk.page_content[:3500],
                    "source": source_name,
                    "page": str(chunk.metadata.get("page", "")),
                },
            }
        )

    for start in range(0, len(payload), UPSERT_BATCH_SIZE):
        batch = payload[start : start + UPSERT_BATCH_SIZE]

        if PINECONE_NAMESPACE:
            index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
        else:
            index.upsert(vectors=batch)

    return len(payload)


def ingest_reports(uploaded_files) -> int:
    raw_documents = load_pdfs(uploaded_files)

    if not raw_documents:
        raise ValueError(
            "No text could be extracted from the uploaded PDF(s). "
            "The report may be scanned or image-only."
        )

    chunks = split_documents(raw_documents)

    if not chunks:
        raise ValueError("The uploaded PDF(s) did not produce usable text chunks.")

    pinecone_client = get_pinecone()
    ensure_index(pinecone_client)

    index = pinecone_client.Index(PINECONE_INDEX_NAME)

    # New upload = new active report session.
    # Removes data from the previously uploaded report(s).
    delete_all_vectors(index)

    return upsert_chunks(index, chunks)


def retrieve(question: str, top_k: int = RETRIEVAL_TOP_K) -> List[Document]:
    embeddings = get_embeddings()
    question_vector = embeddings.embed_query(question)

    query_arguments = {
        "vector": question_vector,
        "top_k": top_k,
        "include_metadata": True,
    }

    if PINECONE_NAMESPACE:
        query_arguments["namespace"] = PINECONE_NAMESPACE

    result = get_index().query(**query_arguments)

    matches = (
        result.get("matches", [])
        if isinstance(result, dict)
        else getattr(result, "matches", []) or []
    )

    documents: List[Document] = []

    for match in matches:
        metadata = (
            match.get("metadata", {})
            if isinstance(match, dict)
            else getattr(match, "metadata", {}) or {}
        )

        text = metadata.get("text", "").strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "filename": metadata.get("source", "uploaded-report"),
                    "source": metadata.get("source", "uploaded-report"),
                    "page": metadata.get("page", ""),
                },
            )
        )

    return documents


def format_context(documents: List[Document]) -> str:
    if not documents:
        return "(No relevant excerpts were found in the currently uploaded report.)"

    context_parts = []

    for number, document in enumerate(documents, start=1):
        source = (
            document.metadata.get("filename")
            or document.metadata.get("source")
            or "uploaded-report"
        )

        page = document.metadata.get("page", "")
        heading = f"[Excerpt {number} | File: {source}"

        if page:
            heading += f" | Page: {page}"

        heading += "]"

        context_parts.append(f"{heading}\n{document.page_content.strip()}")

    return "\n\n---\n\n".join(context_parts)


# ============================================================
# GROQ RESPONSE GENERATION
# ============================================================

@st.cache_resource(show_spinner=False)
def build_llm() -> ChatGroq:
    require_keys()

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        api_key=GROQ_API_KEY,
    )


def extract_text_from_llm_content(content) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(str(block["text"]))

        return "".join(text_parts).strip()

    return str(content).strip()


def answer_as_clinica_ai(question: str, history: list) -> str:
    retrieved_documents = retrieve(question)
    report_context = format_context(retrieved_documents)

    system_prompt = GP_SYSTEM_PROMPT.format(context=report_context)

    messages = [SystemMessage(content=system_prompt)]

    # Conversation helps interpret follow-ups, but report facts must still
    # come from the active report context included in the system prompt.
    for message in history[-MAX_CHAT_HISTORY_MESSAGES:]:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))

    messages.append(HumanMessage(content=question))

    response = build_llm().invoke(messages)

    return extract_text_from_llm_content(response.content)


# ============================================================
# STREAMLIT UI
# ============================================================

def initialize_session_state() -> None:
    defaults = {
        "messages": [],
        "ready": False,
        "last_files": [],
        "chunk_count": 0,
        "error": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_active_report_session() -> None:
    st.session_state.ready = False
    st.session_state.messages = []
    st.session_state.last_files = []
    st.session_state.chunk_count = 0
    st.session_state.error = ""


def render_upload_screen() -> None:
    st.title("Clinica AI — Medical Report Assistant 🩺")
    st.write(
        "Upload one or more medical-report PDFs, then ask questions about the "
        "currently uploaded report."
    )

    uploaded_files = st.file_uploader(
        "Upload medical report PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.session_state.error:
        st.error(st.session_state.error)

    start_chat = st.button(
        "Upload and start chat",
        type="primary",
        disabled=not uploaded_files,
    )

    if not start_chat:
        return

    with st.spinner("Reading reports and preparing the chat..."):
        try:
            chunk_count = ingest_reports(uploaded_files)

            st.session_state.ready = True
            st.session_state.chunk_count = chunk_count
            st.session_state.last_files = [
                uploaded_file.name for uploaded_file in uploaded_files
            ]
            st.session_state.messages = []
            st.session_state.error = ""

        except Exception as error:
            st.session_state.ready = False
            st.session_state.error = f"Could not process the report: {error}"

    st.rerun()


def render_chat_screen() -> None:
    left_column, right_column = st.columns([3, 1])

    with left_column:
        st.title("What would you like to know?")
        active_files = ", ".join(st.session_state.last_files) or "your report"
        st.caption(
            f"Active report session: {active_files} · "
            f"{st.session_state.chunk_count} indexed sections"
        )

    with right_column:
        if st.button("Upload new report"):
            reset_active_report_session()
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about your report...")

    if not question:
        return

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Reviewing the active report..."):
            try:
                answer = answer_as_clinica_ai(
                    question=question,
                    history=st.session_state.messages[:-1],
                )
            except Exception as error:
                answer = f"I could not complete that request: {error}"

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


def main() -> None:
    st.set_page_config(
        page_title="Clinica AI",
        page_icon="🩺",
        layout="centered",
    )

    initialize_session_state()

    if st.session_state.ready:
        render_chat_screen()
    else:
        render_upload_screen()


if __name__ == "__main__":
    main()