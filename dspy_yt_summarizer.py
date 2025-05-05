from langchain_community.chat_models import ChatOllama
from langchain.chains.summarize import load_summarize_chain

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import dspy
import re
import requests

from utils import Summarize

def is_valid_youtube_link(url):
    # Define the regex pattern for YouTube video and channel URLs
    youtube_regex = re.compile(r'^(https?\:\/\/)?(www\.youtube\.com|youtu\.?be)\/.+$')
    
    # Check if the URL matches the regex pattern
    if not youtube_regex.match(url):
        return False

    # Perform a HEAD request to check the validity of the URL
    try:
        response = requests.head(url, allow_redirects=True)
        # YouTube usually returns 200 or 301 status for valid links
        if response.status_code in [200, 301]:
            return True
    except requests.RequestException as e:
        print(f"Request failed: {e}")
    
    return False



def get_video_content(yt_link):
    try:
        loader = YoutubeLoader.from_youtube_url(
            yt_link, add_video_info=False, language=["tr", "en"]
            )
        video_content = loader.load()
        if not video_content:
            raise ValueError("Transcript not available.")
        return video_content
    except Exception as e:
        return f"Retrieving video content failed: {e}"


def split_chunks(video_content):
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=20)
    chunks = splitter.split_documents(video_content)
    return chunks



def summarize_yt_video(yt_link):
    llm = dspy.LM('ollama_chat/qwen3:8b')
    dspy.configure(lm=llm)
    summarizer = Summarize()
    video_content = get_video_content(yt_link)
    chunks = split_chunks(video_content)
    result = summarizer(chunks)
    return result