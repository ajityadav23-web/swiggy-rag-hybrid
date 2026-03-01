# Swiggy Annual Report Hybrid RAG System

## Project Overview
This project implements a **Hybrid Retrieval-Augmented Generation (RAG) system** designed to answer user queries based strictly on information contained in Swiggy’s Annual Report document.

The system combines semantic similarity search with keyword-based ranking to retrieve the most relevant document segments before generating a response.

The objective of this project is to demonstrate practical implementation of document-grounded question answering using modern NLP retrieval architectures.

---

## Key Features
- Hybrid retrieval architecture (Vector + BM25)
- Semantic document chunking
- Context-restricted answer generation
- Vector database indexing
- Interactive web interface
- Fully local execution without paid APIs

---

## Technology Stack
- Python  
- LangChain — retrieval pipeline orchestration  
- Hugging Face — sentence embeddings  
- FAISS — vector similarity search  
- BM25 — lexical retrieval  
- Streamlit — user interface  

---

## Repository Structure
```
swiggy-rag-hybrid/
│── app.py
│── ingest.py
│── rag_pipeline.py
│── requirements.txt
│── README.md
│── data/
│── vectorstore/
```

---

## Installation Instructions

Clone repository:
```bash
git clone https://github.com/ajityadav23-web/swiggy-rag-hybrid.git
cd swiggy-rag-hybrid
```

Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Application

### Step 1 — Process Document
```bash
python ingest.py
```

### Step 2 — Launch Application
```bash
streamlit run app.py
```

---

## Example Query
```
What was the consolidated total income in FY24?
```

---

## Data Source
Swiggy Annual Report 
**Source Link:** *https://www.swiggy.com/corporate/wp-content/uploads/2024/10/Annual-Report-FY-2023-24.pdf*

---

## Repository Link
https://github.com/ajityadav23-web/swiggy-rag-hybrid

---

## Author
**Ajit Yadav**  
Machine Learning Intern Candidate
