# Summarizer

Summarizer is a tool that generates concise summaries of YouTube videos and website content. 
It leverages a local large language model (Qwen 3:8B) via Ollama, and provides an intuitive interface built with Gradio. 

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/summarizer.git
cd summarizer
pip install -r requirements.txt
```
Make sure you have Ollama installed and the Qwen 3:8B model set up locally or make the changes for yourself.

## Usage

After installing the dependencies, you can launch the Summarizer interface by running:

```python
python ui.py
```
This will start a local Gradio web app where you can input a YouTube video link or a website URL to generate a summary using the Qwen 3:8B model via Ollama.
