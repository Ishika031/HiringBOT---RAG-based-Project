import sys
import os

# Fix for module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from core.resume_processor import process_resume
from core.vector_db import get_vector_store

load_dotenv()

st.set_page_config(page_title="Upload Resumes", page_icon="📤", layout="wide")

st.title("📤 Resume Upload Portal")
st.markdown("### Upload candidate resumes → Auto chunked & vectorized with Gemini")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    convert_system_message_to_human=True
)

vector_store = get_vector_store()

uploaded_files = st.file_uploader(
    "Upload PDF Resumes (multiple allowed)", 
    type=["pdf"], 
    accept_multiple_files=True
)

if st.button("🚀 Process & Store All Resumes", type="primary"):
    if uploaded_files:
        progress_bar = st.progress(0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                # Save file
                os.makedirs("resumes", exist_ok=True)
                file_path = f"resumes/{uploaded_file.name}"
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process resume
                docs, metadata = process_resume(file_path, llm)
                
                vector_store.add_documents(docs)

                progress_bar.progress((idx + 1) / len(uploaded_files))
                st.success(f"✅ Processed: **{uploaded_file.name}** → {metadata.name} ({metadata.primary_field})")
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        st.balloons()
        st.success("🎉 All resumes processed and stored successfully!")
        st.info("Now go to the Chatbot page to search for candidates.")
    
    else:
        st.warning("⚠️ Please upload at least one PDF resume.")