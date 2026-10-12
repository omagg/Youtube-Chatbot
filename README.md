# YouTube RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you ask questions about a YouTube video and generates answers grounded in the video's English transcript.

The project uses **Streamlit** for the web interface, **LangChain** to connect the RAG pipeline, **Hugging Face Embeddings** to represent transcript chunks as vectors, **FAISS** for similarity search, and **Groq** to generate answers.

## Features

- Enter a YouTube video URL or video ID.
- Fetch the video's English transcript using `youtube-transcript-api`.
- Split transcript text into overlapping chunks.
- Create embeddings with `google/embeddinggemma-2`.
- Store and search transcript chunks with FAISS.
- Retrieve the four most similar chunks for a question.
- Generate answers with a Groq-hosted chat model.
- Keep the conversation visible during the current Streamlit session.
- Expand an answer's retrieved transcript chunks to inspect the context used.
- Cache the retriever for a video during the app's runtime to avoid rebuilding it on every chat interaction.
- Display progress indicators and user-facing error messages.

> **Current limitations:** The app requests English transcripts only. Videos without an accessible English transcript may not work. Retrieved chunks are shown as text; timestamp-based citations and persistent chat history are not currently implemented.

## How it works

```text
YouTube video URL / ID
          |
          v
Fetch English transcript
          |
          v
Split transcript into chunks
(chunk size: 1000, overlap: 200)
          |
          v
Generate embeddings
(Hugging Face: google/embeddinggemma-2)
          |
          v
Build FAISS vector index
          |
          v
Retrieve top 4 relevant chunks
          |
          v
Prompt + Groq chat model
          |
          v
Answer in the Streamlit chat UI
```

The prompt instructs the model to answer only from the retrieved transcript context and to say it does not know when the context is insufficient. This helps ground answers, but does not guarantee that every answer will be correct.

## Tech stack

- Python
- [Streamlit](https://docs.streamlit.io/)
- [LangChain](https://docs.langchain.com/)
- [Groq](https://console.groq.com/docs/overview)
- [Hugging Face](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)
- [python-dotenv](https://github.com/theskumar/python-dotenv)

## Prerequisites

- Python installed on your system.
- A Groq API key.
- Internet access to fetch transcripts and access the Groq API.
- Enough local resources to download and run the selected embedding model.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/omagg/Youtube-Chatbot.git
cd Youtube-Chatbot
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the packages used by the application:

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Get an API key from the [Groq Console](https://console.groq.com/).


### 5. Run the app

If your Streamlit application is named `main.py`:

```bash
python -m streamlit run main.py
```

## Usage

1. Open the app in your browser.
2. Paste a YouTube URL or an 11-character video ID into the sidebar.
3. Click **Load video** and wait for the transcript to be processed and indexed.
4. Type a question into the chat input.
5. Read the answer and expand **View retrieved transcript chunks** to inspect the retrieved text.
6. Load another video from the sidebar to start a new conversation for that video.

Example questions:

- What is the main idea of this video?
- Can you summarize the video?
- Which methods or techniques are discussed?
- What does the speaker say about a particular topic?

Answers depend on the transcript content and the chunks retrieved for the question.

## Configuration

The current implementation uses these settings:

| Setting | Current value |
|---|---|
| Transcript language | English (`en`) |
| Chunk size | 1000 characters |
| Chunk overlap | 200 characters |
| Number of retrieved chunks | 4 |
| Embedding model | `google/embeddinggemma-2` |
| Groq model | `openai/gpt-oss-120b` |
| Temperature | `0.2` |
| Vector store | FAISS |

These are the values in the current application code. Verify that the selected Groq model is available to your account and supported by the current Groq API.

## Troubleshooting

### `GROQ_API_KEY` is missing

- Confirm that `.env` is in the project root.
- Check that the variable is named exactly `GROQ_API_KEY`.
- Restart Streamlit after changing environment variables.
- Do not share your API key in an issue or public repository.

### No English transcript is available

Some videos have captions disabled or do not provide English captions. Try another video with accessible English captions. The current version does not automatically translate transcripts.

### Transcript fetching fails

Check your internet connection and the video's availability. The transcript service may also fail because of access restrictions or changes on YouTube. Inspect the technical error shown in the app for more detail.

### Embedding model download or initialization fails

Check your internet connection, installed dependencies, and whether the model is accessible in your environment. The first run may take longer because model files may need to be downloaded.

### FAISS or package import errors

Ensure the virtual environment is activated and install the dependencies listed above into that same environment. If errors persist, record your Python and package versions and test a compatible set of versions.

### Answers are incomplete or incorrect

RAG quality depends on transcript quality, chunk boundaries, embedding performance, and retrieval. Inspect the retrieved chunks before changing the model or prompt. The application is a learning project and its responses should be verified against the video when accuracy matters.

## Project structure

A simple setup can look like this:

```text
youtube-rag-chatbot/
├── main.py
├── .env                 
├── .gitignore
├── requirements.txt     
└── README.md
```

## Security and privacy

- Keep API keys in environment variables or a local `.env` file.
- Do not commit secrets to GitHub.
- Transcript text is sent as context to the configured Groq model when answering questions.
- Review the relevant service terms and privacy policies before using private or sensitive video content.

## Possible future improvements

- Add transcript timestamps and links to relevant moments in the video.
- Support additional transcript languages where available.
- Add a curated evaluation dataset and measure retrieval and answer quality.
- Experiment with chunk sizes, overlap, embedding models, and retrieval strategies.
- Add streaming responses and suggested questions.
- Add persistent conversation history and a way to manage multiple videos.
- Add automated tests and pin dependency versions.

## Documentation

- [Streamlit chat app tutorial](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps)
- [Streamlit Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)
- [LangChain documentation](https://docs.langchain.com/)
- [LangChain embedding integrations](https://docs.langchain.com/oss/python/integrations/embeddings)
- [Groq documentation](https://console.groq.com/docs/overview)
- [YouTube Transcript API repository](https://github.com/jdepoix/youtube-transcript-api)

## Author
Om Aggarwal
