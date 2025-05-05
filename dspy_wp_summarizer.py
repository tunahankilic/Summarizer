from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import dspy
import requests
from utils import Summarize


def is_accessible(url):
    """
    Check if the webpage link is valid by making an HTTP request.

    Parameters:
    url (str): The URL of the webpage to check.

    Returns:
    bool: True if the URL is accessible (status code 200), False otherwise.
    """
    try:
        response = requests.get(url, timeout=10)
        # Check if the status code indicates success (200 OK)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.RequestException as e:
        # Catch any request exceptions (e.g., connection errors, timeouts)
        print(f"Error accessing {url}: {e}")
        return False


def get_webpage_content(url):
    try:
        loader = WebBaseLoader(url)
        webpage_content = loader.load()
        if not webpage_content:
            raise ValueError("Webpage is not available.")
        return webpage_content
    except Exception as e:
        return f"Retrieving video content failed: {e}"



def split_chunks(webpage_content):
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=20)
    chunks = splitter.split_documents(webpage_content)
    return chunks



def summarize_webpage(wp_link):
    llm = dspy.LM('ollama_chat/qwen3:8b')
    dspy.configure(lm=llm)
    summarizer = Summarize()
    webpage_content = get_webpage_content(wp_link)
    chunks = split_chunks(webpage_content)
    result = summarizer(chunks)
    return result