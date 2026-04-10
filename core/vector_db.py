import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    persist_dir = "vector_store"
    os.makedirs(persist_dir, exist_ok=True)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="candidate_resumes"
    )