import dspy


class SummarizeSignature(dspy.Signature):
    """
        Craft a Detailed Summary: Ensure the summary is thorough, in-depth, and complex, while maintaining clarity.
        Incorporate Main Ideas: Focus on the main ideas and essential information, eliminating extraneous language and highlighting critical aspects.
        Rely on Provided Text: Base the summary strictly on the provided text, avoiding the inclusion of external information.
        Format for Clarity: Present the summary in paragraph form to ensure it is easy to understand.
        Output Only the Summary: Ensure the response contains only the summary text without any introductory or concluding remarks.

    """

    text = dspy.InputField(desc="a text to summarize")
    summary: str = dspy.OutputField(min_length=256,
        desc="a summary of a given text")





class Summarize(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought(SummarizeSignature)

    def forward(self, text: str):
        summary = self.summarize(
            text=text
        )
        return summary