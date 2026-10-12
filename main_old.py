from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

#functions:=>

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text


#document ingestion
video_id = "1LRvMRC_pfY" # only the ID, not full URL
ytt_api = YouTubeTranscriptApi()
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = ytt_api.fetch(video_id, languages=['en'])

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)

except TranscriptsDisabled:
    print("No captions available for this video.")

#text splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

#embedding genr.
embeddings = HuggingFaceEmbeddings(model_name="google/embeddinggemma-2")

#vector store
vector_store = FAISS.from_documents(chunks, embeddings)

#retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

#augmentation
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the implementation part of muse discussed in this video?"
retrieved_docs    = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})

#generation
answer = llm.invoke(final_prompt)

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

print(main_chain.invoke('Can you summarize the video'))