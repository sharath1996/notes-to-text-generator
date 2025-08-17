from textwrap import dedent
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os


class ImagesToMarkdownInput(BaseModel):
    list_images: list[str] = Field(..., description="List of image contents to be converted to Markdown")

class ImagesToMarkdownOutput(BaseModel):
    str_markdown: str = Field(..., description="Markdown representation of the images")
    
class ImagesToMarkdown:

    def __init__(self):
        ...
    
    def convert(self, param_obj_input: ImagesToMarkdownInput) -> ImagesToMarkdownOutput:
        """
        Converts images in the specified input path to a Markdown format.
        
        :param param_obj_input: Input parameters containing the path to the images.
        :return: Output containing the Markdown representation of the images.
        """
        local_str_overallMarkdown = ""
        for local_str_imageData in param_obj_input.list_images:
                local_obj_output = self.convert_to_text(local_str_imageData)
                local_str_overallMarkdown += local_obj_output.str_markdown + "\n"

        return ImagesToMarkdownOutput(str_markdown=local_str_overallMarkdown.strip())

    def convert_to_text(self, param_str_imageData:str)-> ImagesToMarkdownOutput:
        """
        Converts image data to a text representation.
        
        :param param_str_imageData: Base64 encoded image data.
        :return: Text representation of the image.
        """
        
        local_obj_azureChatOpenAI = AzureChatOpenAI(azure_deployment= os.getenv("AZURE_DEPLOYEMENT_GENERIC"))
        local_obj_systemMessage = SystemMessage(
            content= dedent("""
            You are a specialized assistant designed to convert images of handwritten notes into clean, readable Markdown format. Follow these instructions precisely:
            Input
            You will receive one or more images containing handwritten notes.
            These notes may include text, mathematical formulas, tables, and block diagrams.
            Output Format
            - Plain Markdown: Convert all textual content into plain Markdown without any special formatting (e.g., no bold, italics, headers, or links unless explicitly present in the notes).
            - Formulas: Preserve all mathematical formulas using LaTeX syntax wrapped in double dollar signs ($$...$$) for block formulas or single dollar signs (\$...\$) for inline formulas.
            - Tables: Convert any tabular data into Markdown table format using pipes (|) and dashes (-) to define rows and columns.
            - Block Diagrams: If any block diagrams are present, convert them into Mermaid syntax using appropriate diagram types (e.g., flowchart, sequence diagram). Use best-effort interpretation based on the visual structure.
            Rules
            - Do not apply any stylistic enhancements or formatting beyond what is specified.
            - Maintain the original structure and order of the content as closely as possible.
            - If any content is unclear or ambiguous, make a best-effort guess and annotate it with a comment in Markdown (<!-- unclear -->).
            You should only return the Markdown content without any additional explanations or comments.
            In case the page is emtp, return an empty string.
            """)
        )

        local_obj_humanMessage = HumanMessage(
            content = [
                {
                "type": "text",
                "text" : "Convert the following image data to Markdown format.",
                },
                {
                "type": "image",
                "source_type": "base64",
                "mime_type": "image/png",  # or image/png, etc.
                "data": param_str_imageData,
                }
            ])
        
        local_list_messages = [local_obj_systemMessage, local_obj_humanMessage]
        local_obj_response = local_obj_azureChatOpenAI.with_structured_output(ImagesToMarkdownOutput).invoke(local_list_messages)
        return local_obj_response
        
