import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain

from core.vector_db import get_vector_store
from core.email_sender import send_emails_to_candidates

load_dotenv()

st.set_page_config(page_title="HiringBot - AI Recruiter", page_icon="🤖", layout="wide")

st.title("🤖 HiringBOT")
st.caption("Conversational AI Recruiter • Smart Filters • Candidate Selection • Email Agent")

# ====================== SESSION STATE ======================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hey there! 👋 How can I help you today?"}]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_candidates" not in st.session_state:
    st.session_state.current_candidates = []
if "selected_candidates" not in st.session_state:
    st.session_state.selected_candidates = set()
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    convert_system_message_to_human=True
)

vector_store = get_vector_store()

# Strict Prompt for better conversation
SYSTEM_PROMPT = """You are a helpful and accurate AI recruiter assistant.
Always base your answers only on the retrieved candidate context.
Be conversational and natural.
If candidates are found, list them clearly.
Never hallucinate candidate names or details.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

document_chain = create_stuff_documents_chain(llm, prompt)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ====================== CHAT INPUT ======================
if prompt_input := st.chat_input("Ask anything about candidates... (e.g. 'Java candidates', 'Show Python developers in Nagpur', 'Tell me more about Ishika')"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        lower = prompt_input.lower().strip()

        # Email confirmation
        if st.session_state.selected_candidates and lower in ["yes", "y", "sure", "yes please", "ok", "please send", "send emails", "send"]:
            selected_list = [c for c in st.session_state.current_candidates if c.get("email") in st.session_state.selected_candidates]
            if selected_list:
                with st.spinner("Sending emails..."):
                    count = send_emails_to_candidates(selected_list, st.session_state.last_query, llm)
                response = f"✅ Emails sent successfully to {count} candidates!"
                st.session_state.selected_candidates.clear()
            else:
                response = "⚠️ No candidates selected."
        else:
            print("\n" + "="*80)
            print(f"QUERY: {prompt_input}")
            print("="*80)

            # Always do similarity search with score for transparency
            scored_docs = vector_store.similarity_search_with_score(prompt_input, k=12)

            filtered_docs = []
            print("RELEVANCE SCORES (Lower = Better | Max allowed = 0.65)")
            print("-" * 85)

            for i, (doc, score) in enumerate(scored_docs, 1):
                name = doc.metadata.get("name", "Unknown")
                field = doc.metadata.get("primary_field", "N/A")
                print(f"{i:2d}. {name:<32} | {field:<28} | Score: {score:.4f}", end="")

                if score <= 0.69:
                    print("  → ✅ KEPT")
                    filtered_docs.append(doc)
                else:
                    print("  → ❌ FILTERED")

            print(f"\nKept {len(filtered_docs)} candidate(s) with score ≤ 0.65\n")

            context_docs = filtered_docs

            if context_docs:
                unique_cands = {}
                for doc in context_docs:
                    meta = doc.metadata
                    if meta.get("email"):
                        email = meta["email"]
                        if email not in unique_cands:
                            unique_cands[email] = {
                                "name": meta.get("name", "N/A"),
                                "email": email,
                                "primary_field": meta.get("primary_field", "N/A"),
                                "experience_years": meta.get("experience_years"),
                                "location": meta.get("location", "N/A"),
                                "full_resume_path": meta.get("full_resume_path")
                            }

                st.session_state.current_candidates = list(unique_cands.values())
                st.session_state.last_query = prompt_input

                if not st.session_state.selected_candidates:
                    st.session_state.selected_candidates = {c["email"] for c in st.session_state.current_candidates}

                response = f"I found {len(st.session_state.current_candidates)} relevant candidate(s) for your query."
            else:
                st.session_state.current_candidates = []
                st.session_state.selected_candidates.clear()
                response = "Sorry, I couldn't find any candidates with good relevance score (≤ 0.65). Can you try rephrasing your query?"

        placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.extend([HumanMessage(content=prompt_input), AIMessage(content=response)])

    st.rerun()

# ====================== CANDIDATE LIST ======================
if st.session_state.current_candidates:
    st.divider()
    st.subheader(f"📋 Matching Candidates ({len(st.session_state.current_candidates)})")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        if st.button("Select All"):
            st.session_state.selected_candidates = {c["email"] for c in st.session_state.current_candidates}
            st.rerun()
    with col3:
        if st.button("Deselect All"):
            st.session_state.selected_candidates.clear()
            st.rerun()

    for cand in st.session_state.current_candidates:
        email = cand.get("email")
        if not email: continue

        c1, c2, c3 = st.columns([0.6, 7.0, 1.5])
        with c1:
            checked = st.checkbox("Select", value=email in st.session_state.selected_candidates,
                                  key=f"check_{email}", label_visibility="hidden")
            if checked:
                st.session_state.selected_candidates.add(email)
            else:
                st.session_state.selected_candidates.discard(email)

        with c2:
            st.markdown(f"**{cand.get('name', 'N/A')}** — {cand.get('primary_field', 'N/A')} • "
                        f"{cand.get('experience_years', 'N/A')} yrs • {cand.get('location', 'N/A')}")

        with c3:
            path = cand.get("full_resume_path")
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button("⬇️ Download", data=pdf_bytes, file_name=os.path.basename(path),
                                   mime="application/pdf", key=f"dl_{email}")
            else:
                st.write("—")

    if st.session_state.selected_candidates:
        if st.button(f"✉️ Send Emails to {len(st.session_state.selected_candidates)} Selected",
                     type="primary", use_container_width=True):
            st.info("Type **yes** in the chat input to confirm sending emails.")