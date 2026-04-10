import os
# Updated imports for newer LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .models import CandidateMetadata
from .prompts import EXTRACTION_PROMPT

def process_resume(file_path: str, llm):
    """Parse PDF → extract structured metadata → chunk → return Documents"""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])

        # Extract structured metadata using Gemini
        structured_llm = llm.with_structured_output(CandidateMetadata)
        chain = EXTRACTION_PROMPT | structured_llm
        metadata = chain.invoke({
            "resume_text": full_text, 
            "format_instructions": CandidateMetadata.model_json_schema()
        })

        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = splitter.split_text(full_text)

        # Create LangChain Documents
        documents = []
        filename = os.path.basename(file_path)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    **metadata.model_dump(),
                    "source": filename,
                    "chunk_index": i,
                    "full_resume_path": file_path
                }
            )
            documents.append(doc)

        return documents, metadata

    except Exception as e:
        raise Exception(f"Failed to process resume {file_path}: {str(e)}")