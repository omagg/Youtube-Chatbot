import os
import re
from urllib.parse import parse_qs, urlparse

import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="▶️",
    layout="wide",
)

st.title("▶️ YouTube RAG Chatbot")
st.caption("Ask questions about a YouTube video and get answers grounded in its transcript.")


def extract_video_id(value: str) -> str:
    """Accept an 11-character YouTube video ID or a common YouTube URL."""
    value = value.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id:
            video_id = query_id
        else:
            parts = [part for part in parsed.path.split("/") if part]
            video_id = parts[1] if len(parts) >= 2 and parts[0] in {
                "shorts", "embed", "live"
            } else None
    else:
        video_id = None

    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Enter a valid YouTube video URL or 11-character video ID.")

    return video_id


@st.cache_resource(show_spinner=False)
def build_retriever(video_id: str):
    """Fetch a transcript and build a reusable FAISS retriever for this video."""
    transcript_api = YouTubeTranscriptApi()
    transcript_segments = transcript_api.fetch(video_id, languages=["en"])
    transcript = " ".join(segment.text for segment in transcript_segments).strip()

    if not transcript:
        raise ValueError("The English transcript for this video is empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(
        model_name="google/embeddinggemma-2"
    )
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    """Create one reusable Groq chat model client."""
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file and restart the app."
        )

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
    )


def format_docs(documents) -> str:
    return "\n\n".join(document.page_content for document in documents)


prompt = PromptTemplate(
    template="""You are a helpful assistant answering questions about a YouTube video.
Answer using ONLY the provided transcript context.
If the context does not contain enough information, say that you don't know.
Do not invent details.

Transcript context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)


with st.sidebar:
    st.header("Video settings")
    with st.form("video_form"):
        video_input = st.text_input(
            "YouTube URL or video ID",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a YouTube video URL or its 11-character video ID.",
        )
        load_video = st.form_submit_button("Load video", type="primary", use_container_width=True)

    st.divider()
    st.caption("Model settings")
    st.write("**Chat model:** Groq")
    st.write("**Embeddings:** Google EmbeddingGemma 2")
    st.write("**Vector store:** FAISS")
    st.write("**Retrieved chunks:** 4")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if "active_video_id" not in st.session_state:
    st.session_state.active_video_id = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if load_video:
    try:
        selected_id = extract_video_id(video_input)
        if selected_id != st.session_state.active_video_id:
            st.session_state.messages = []
        st.session_state.active_video_id = selected_id
        st.session_state.video_load_error = ""
    except ValueError as exc:
        st.session_state.video_load_error = str(exc)

if st.session_state.get("video_load_error"):
    st.error(st.session_state.video_load_error)

video_id = st.session_state.active_video_id

if not video_id:
    st.info("Load a YouTube video from the sidebar to start chatting.")
    st.markdown(
        """
        **How to use**
        1. Paste a YouTube video URL in the sidebar.
        2. Select **Load video** and wait for transcript indexing to finish.
        3. Ask a question in the chat box below.
        """
    )
    st.stop()

st.video(f"https://www.youtube.com/watch?v={video_id}")

try:
    with st.spinner("Loading transcript and preparing the search index..."):
        retriever = build_retriever(video_id)
except Exception as exc:
    st.error(
        "Couldn't prepare this video's transcript. It may have captions disabled, "
        "no English transcript, or the transcript service may be temporarily unavailable."
    )
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

# Render previous messages on each Streamlit rerun.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View retrieved transcript chunks"):
                for index, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**Chunk {index}**")
                    st.write(source)

question = st.chat_input("Ask something about this video...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the transcript and generating an answer..."):
                retrieved_docs = retriever.invoke(question)
                context = format_docs(retrieved_docs)
                llm = get_llm()
                answer_chain = prompt | llm | StrOutputParser()
                answer = answer_chain.invoke(
                    {"context": context, "question": question}
                )

            st.markdown(answer)

            with st.expander("View retrieved transcript chunks"):
                for index, document in enumerate(retrieved_docs, start=1):
                    st.markdown(f"**Chunk {index}**")
                    st.write(document.page_content)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": [document.page_content for document in retrieved_docs],
                }
            )
        except Exception as exc:
            st.error("Something went wrong while generating the answer.")
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")
