from langchain_community.chat_models import ChatOllama
from langchain.chains.summarize import load_summarize_chain

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import re
import requests


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



def summarization_chain():
        prompt_template = PromptTemplate(
        template="""As a professional summarizer, create a detailed and comprehensive summary of the provided text, be it an article, post, conversation, or passage, while adhering to these guidelines:
        
        Craft a Detailed Summary: Ensure the summary is thorough, in-depth, and complex, while maintaining clarity.
        Incorporate Main Ideas: Focus on the main ideas and essential information, eliminating extraneous language and highlighting critical aspects.
        Rely on Provided Text: Base the summary strictly on the provided text, avoiding the inclusion of external information.
        Format for Clarity: Present the summary in paragraph form to ensure it is easy to understand.
        Output Only the Summary: Ensure the response contains only the summary text without any introductory or concluding remarks.
        
        By following this optimized prompt, you will generate an effective summary that encapsulates the essence of the given text in a clear, detailed, and reader-friendly manner. Optimize the output as a markdown file.
        
        "{text}"

        DETAILED SUMMARY:""",
        input_variables=["text"],
        #output_parser=StrOutputParser()
    )
        llm = ChatOllama(model="llama3.1:8b-instruct-q4_0", base_url="http://127.0.0.1:11434")
        summary_chain = load_summarize_chain(llm=llm, prompt=prompt_template, verbose=False)
        return summary_chain



def summarize_webpage(wp_link):
    webpage_content = get_webpage_content(wp_link)
    chunks = split_chunks(webpage_content)
    summary_chain = summarization_chain()
    result = summary_chain.run(chunks)
    return result