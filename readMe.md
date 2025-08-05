# GitHub + PDF Analyzer – Setup Guide

### 1. Prerequisites

- Python 3.9 – 3.11 (tested on 3.11)
- `pip` / `venv`
- Git (optional, for repo cloning)
- Pinecone account (serverless project, e.g., `aws■us■east■1`)
- Google Generative AI key (optional, enables Gemini answers)

### 2. Clone / Unzip

```bash
$ unzip repo_analyser.zip -d repo_analyser
$ cd repo_analyser

3. Create Virtual Environment & Install Dependencies

```bash

$ python3 -m venv venv
$ source venv/bin/activate    # Windows: venv\Scripts\activate
$ pip install -r requirements.txt


4. Set Environment Variables

```bash
PINECONE_API_KEY=<your_key>
GOOGLE_API_KEY=<optional>
export TOKENIZERS_PARALLELISM=false  # silence HF tokenizers warning

$ streamlit run streamlit_app.py