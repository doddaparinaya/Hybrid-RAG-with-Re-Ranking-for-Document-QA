# llm.py
from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyAk5d22UqL7GvQeP0mPjjp0vtIF09eKghw"

def get_llm_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )