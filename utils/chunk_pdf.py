# utils/chunk_pdf.py

import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf_files(uploaded_pdfs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = []
    for pdf in uploaded_pdfs:
        data = pdf.read()
        doc  = fitz.open(stream=data, filetype="pdf")
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text().strip()
            if not text:
                continue
            path = f"{pdf.name} [page {i+1}]"
            for chunk in splitter.split_text(text):
                docs.append(Document(page_content=chunk, metadata={"path": path}))
    return docs
