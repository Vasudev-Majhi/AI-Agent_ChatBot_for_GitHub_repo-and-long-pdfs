import os, traceback, time
import streamlit as st
from dotenv import load_dotenv

from utils.chunk_github import process_github_repo
from utils.chunk_pdf    import process_pdf_files
from utils.rag_pipeline import setup_rag_chain, run_query

load_dotenv()
st.set_page_config(page_title="Codebase Analyzer", layout="centered")
st.title("📦 GitHub + PDF Analyzer")

# input
repo_url = st.text_input("🔗 GitHub Repo URL")
uploaded_pdfs = st.file_uploader("📄 Upload PDF Files", type="pdf", accept_multiple_files=True)

# cache loaders & store
@st.cache_data(ttl=24*3600)
def get_github_docs(url): return process_github_repo(url)
@st.cache_data(ttl=24*3600)
def get_pdf_docs(files): return process_pdf_files(files)
@st.cache_resource
def build_index(docs): return setup_rag_chain(docs)

if st.button("🚀 Process"):
    if not repo_url:
        st.error("Enter a GitHub URL.")
    else:
        try:
            with st.spinner("Cloning & chunking…"):
                gh_docs = get_github_docs(repo_url)
            with st.spinner("Reading PDFs…"):
                pdf_docs = get_pdf_docs(tuple(uploaded_pdfs))
            with st.spinner("Building/Retrieving index…"):
                vs = build_index(gh_docs + pdf_docs)
            st.session_state.vs = vs
            st.success("✅ Ready!")
        except Exception as e:
            st.error("Processing failed")
            st.text(traceback.format_exc())

if "vs" in st.session_state:
    query = st.text_input("💬 Ask a question")
    if st.button("🔍 Submit Query") and query:
        with st.spinner("Generating answer…"):
            try:
                ans = run_query(st.session_state.vs, query)
                st.markdown("### 🧠 Answer")
                st.write(ans)
            except Exception:
                st.error("Query failed")
                st.text(traceback.format_exc())
