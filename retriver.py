# retriever.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, documents):
        self.docs = documents
        self.texts = [doc.page_content for doc in documents]

        # Dense
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectordb = FAISS.from_documents(documents, self.embedding)

        # BM25
        tokenized = [text.split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, k=5):
        # Dense
        dense_docs = self.vectordb.similarity_search(query, k=k)

        # BM25
        scores = self.bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[-k:]
        bm25_docs = [self.docs[i] for i in top_idx]

        # Merge
        combined = list({doc.page_content: doc for doc in dense_docs + bm25_docs}.values())

        return combined