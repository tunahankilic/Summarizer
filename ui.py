import gradio as gr

from youtube_summarizer import summarize_yt_video, is_valid_youtube_link
from webpage_summarizer import summarize_webpage, is_accessible


def summarize(url):
    if is_valid_youtube_link(url):
        result = summarize_yt_video(url)
        return result
    elif is_accessible(url):
        result = summarize_webpage(url)
        return result
    else:
        return "Invalid or inaccessible URL. Please enter a valid URL."



with gr.Blocks() as demo:
    gr.Markdown("# Website and Youtube Summarizer")

    with gr.Row():
        with gr.Column():
            url = gr.Textbox(label='URL', placeholder="Enter URL here")

            with gr.Row():
                btn_summarize = gr.Button("Summarize", variant='primary')
                btn_clear = gr.ClearButton(components=[url])

            output = gr.Textbox(label="Output", placeholder="Summary will appear here", lines=10)

            btn_summarize.click(summarize, inputs=[url], outputs=[output])
            btn_clear.click()


demo.launch()
