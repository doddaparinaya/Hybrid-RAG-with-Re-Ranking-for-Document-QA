# app.py
from ingest import loading_and_spliting_pdf
from retriver import HybridRetriever
from chains import get_query_rewriter
from reranker import reranking_chunks as rerank
from evaluator import quality_of_retrieval as check_retrieval_quality
from load_llm import get_llm_model
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load
docs = loading_and_spliting_pdf("Attention is All you Need.pdf")

# Init

llm = get_llm_model()
retriever = HybridRetriever(docs)
rewriter = get_query_rewriter(llm)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ Direct LLM call
def generate_answer(llm, context, question):
    prompt = f"""
Answer ONLY from the given context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""
    response = llm.invoke(prompt)
    return response.content


def rag_pipeline(query):
    # 1. Rewrite query
    refined_query = rewriter(query)

    # 2. Retrieve
    docs = retriever.retrieve(refined_query)

    # 3. Quality Check
    quality, score = check_retrieval_quality(
        embedding_model,
        refined_query,
        docs
    )

    if quality == "LOW":
        return f"⚠️ Low retrieval quality ({score:.2f}). Try rephrasing."

    # 4. Re-rank
    ranked_docs = rerank(refined_query, docs)

    # 5. Build context
    context = "\n\n".join([doc.page_content for doc in ranked_docs[:3]])

    # 6. Generate answer (NO LLMChain)
    answer = generate_answer(llm, context, query)

    return answer


# CLI loop
if __name__ == "__main__":
    q = input("Ask: ")
    print(rag_pipeline(q))