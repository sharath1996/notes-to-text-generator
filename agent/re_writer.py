from textwrap import dedent
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os

class SectionRewriteInput(BaseModel):
    """
    Input model for the section rewrite process.
    """
    str_nameOftheSection: str = Field(..., description="Name of the section to be rewritten")
    str_markdown: str = Field(..., description="Markdown representation of the section to be rewritten")

class SectionRewriteOutput(BaseModel):
    """
    Output model for the section rewrite process.
    """
    str_sectionName: str = Field(..., description="Name of the rewritten section")
    str_rewrittenSection: str = Field(..., description="Rewritten section content in Markdown format")

class SectionRewriter:

    def __init__(self):

        self._obj_llm = AzureChatOpenAI(
            azure_deployment=os.environ.get("DEPLOYEMENT_REASONING"),
        )
    
    def rewrite(self, param_obj_input:SectionRewriteInput)-> SectionRewriteOutput:
        """
        Rewrite the section based on the provided input.
        
        :param param_obj_input: Input data containing section name and markdown content.
        :return: Output data containing rewritten section name and content.
        """
        
        local_obj_systemPrompt = SystemMessage(content=dedent("""
        You are an expert at transforming hand-written notes into clear, engaging, and high-quality textbook sections. You will receive the name of a section along with hand-written notes for that section (transcribed for clarity).
        Your responsibilities are as follows:
        Rewrite the content in a polished, textbook-quality manner, suitable for an advanced-level audience.
        Thoroughly cover all topics and points present in the original hand-written notes.
        Add any necessary explanations, clarifications, or relevant examples that would help the reader understand the material, while ensuring all original information is included.
        Make the section engaging and interesting to read, using an instructive and accessible tone.
        Use markdown formatting for your response.
        Add more examples, explanations, and clarifications as needed to ensure the content is comprehensive and clear.
        Ensure that all equations, code blocks, and diagrams are correctly formatted and syntactically correct.
        Correct any errors in equations (including LaTeX), code blocks, or diagrams (such as mermaid diagrams), ensuring all content is properly formatted and syntactically correct.
        Do not change or paraphrase the section name.
        Begin your rewritten content at the second heading level (##), without including the section name itself in the text.
        Include only information from the provided hand-written notes; do not add unrelated content or external context.                                        
        """))

        local_obj_humanPrompt = HumanMessage(
            content=dedent(f"""
            Section Name: {param_obj_input.str_nameOftheSection}
            Markdown Notes: {param_obj_input.str_markdown}
            """)
        )

        local_obj_response = self._obj_llm.with_structured_output(SectionRewriteOutput).invoke([local_obj_systemPrompt, local_obj_humanPrompt])
    

        return local_obj_response