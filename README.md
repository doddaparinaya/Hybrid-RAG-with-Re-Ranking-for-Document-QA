# Hybrid RAG with Re-Ranking for Document QA

A high-performance **PDF Question Answering system** built using **Hybrid Retrieval-Augmented Generation (RAG)**.  
This project combines **BM25 + Dense Retrieval**, **Cross-Encoder Re-Ranking**, and **LLM-based response generation** to deliver accurate and context-aware answers from documents.

## Overview

Traditional RAG systems often suffer from poor retrieval quality.  
This project improves performance by introducing:

- Query rewriting
- Hybrid retrieval (BM25 + embeddings)
- Retrieval quality validation
- Cross-encoder re-ranking

## Architecture
User Query
↓
Query Rewriting (LLM)

↓
Hybrid Retrieval (BM25 + Dense)

↓
Retrieval Quality Check

↓
Re-Ranking (Cross Encoder)

↓
Top-K Context Selection

↓
LLM (Google Gemini)

↓
Final Answer

## Features

- 📄 PDF-based document chat
- 🔍 Hybrid retrieval (semantic + keyword)
- 🧠 Query rewriting using LLM
- 🏆 Cross-encoder re-ranking
- 📊 Retrieval quality evaluation
- ⚡ Fast similarity search using FAISS
- 🤖 Google Gemini for answer generation

## Tech Stack

- **LLM**: Google Gemini
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector DB**: FAISS
- **Keyword Search**: BM25 (`rank_bm25`)
- **Re-Ranking**: Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
- **PDF Processing**: PyMuPDF
- **Language**: Python
