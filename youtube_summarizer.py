from langchain_community.chat_models import ChatOllama
from langchain.chains.summarize import load_summarize_chain

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


import re
import requests

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



def summarization_chain():
    prompt_template = PromptTemplate(
        template="""As a professional summarizer specialized in video content, create a detailed and comprehensive summary of the YouTube video transcript provided. Follow these guidelines closely:
        
        Capture the Essence: Focus on the main ideas and key details, ensuring the summary is in-depth and insightful. Reflect any narrative or instructional elements present in the video.
        Enhance Clarity: Exclude redundant expressions and non-critical details to enhance clarity and conciseness.
        Stick to the Transcript: Base the summary strictly on the provided transcript, avoiding assumptions or additions from external sources.
        Structured Presentation: Present the summary in well-structured paragraph form, making it easy to read and understand.
        Conclude Properly: End with "[End of Notes, Message #X]", where "X" is the sequence number of the summarizing request, to indicate task completion.
        Output Only the Summary: Ensure the response contains only the summary text without any introductory or concluding remarks.

        
        "{text}"

        DETAILED SUMMARY:""",
        input_variables=["text"],
        output_parser=StrOutputParser()
    )
    llm = ChatOllama(model="llama3.1:8b-instruct-q4_0", base_url="http://127.0.0.1:11434")
    summary_chain = load_summarize_chain(llm=llm, prompt=prompt_template, verbose=False)
    return summary_chain



def summarize_yt_video(yt_link):
    video_content = get_video_content(yt_link)
    chunks = split_chunks(video_content)
    summary_chain = summarization_chain()
    result = summary_chain.run(chunks)
    return result