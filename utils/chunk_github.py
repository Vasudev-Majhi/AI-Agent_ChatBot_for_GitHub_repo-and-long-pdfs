# utils/chunk_github.py

import os
import shutil
from git import Repo
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_github_repo(
    url,
    repo_path="temp_repo",
    chunk_size=500,
    chunk_overlap=50,
    exclude_dirs=None,
    max_file_size=1_000_000,
):
    exclude_dirs = exclude_dirs or {".git", "node_modules", "__pycache__"}
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    Repo.clone_from(url, repo_path, depth=1, single_branch=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for fn in files:
            if not fn.endswith((".py", ".md", ".js", ".ts",)):
                continue
            full = os.path.join(root, fn)
            if os.path.getsize(full) > max_file_size:
                continue
            try:
                text = open(full, encoding="utf8", errors="ignore").read()
            except:
                continue
            rel = os.path.relpath(full, repo_path)
            for chunk in splitter.split_text(text):
                docs.append(Document(page_content=chunk, metadata={"path": rel}))
    return docs
