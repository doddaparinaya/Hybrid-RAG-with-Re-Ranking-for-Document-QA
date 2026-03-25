# chains.py
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

# def get_query_rewriter(llm):
#     prompt = PromptTemplate(
#         input_variables=["query"],
#         template="""
# Rewrite the user query to improve retrieval.
# Make it specific and keyword-rich.

# Query: {query}
# """
#     )
#     return LLMChain(llm=llm, prompt=prompt)
# chains.py

def get_query_rewriter(llm):

    def rewrite(query):
        prompt = f"""
Rewrite the user query to improve retrieval.
Make it specific and keyword-rich.

Query: {query}
"""
        response = llm.invoke(prompt)

        # Gemini returns AIMessage → extract text
        return response.content.strip()

    return rewrite