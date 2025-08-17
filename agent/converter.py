from .pages_to_images import PagesToImages, PagesToImagesInput
from .images_to_markdown import ImagesToMarkdown, ImagesToMarkdownInput
from .section_generator import SectionGenerator, SectionGeneratorInput
from .re_writer import SectionRewriteInput, SectionRewriter
from pydantic import BaseModel, Field

class HandwrittenNotesConverterInput(BaseModel):
    """
    Input model for the handwritten notes conversion process.
    """
    str_inputPath: str = Field(..., description="Path to the input PDF file containing handwritten notes")
    str_outputPath: str = Field(..., description="Path to save the output Markdown file")

class HandwrittenNotesConverter:
    """
    A class to convert handwritten notes into Markdown format.
    """

    def __init__(self):
        ...
        

    def convert(self, param_obj_input:HandwrittenNotesConverterInput) -> None:
        """
        Convert handwritten notes images to Markdown format.
        """
        # Step 1: Convert PDF pages to images
        local_obj_converter = PagesToImages()
        local_obj_images_input = PagesToImagesInput(
            str_inputPath=param_obj_input.str_inputPath,
            bool_saveImages=False
        )
        local_obj_imagesOutput = local_obj_converter.convert(local_obj_images_input)

        # Step 2: Convert images to Markdown
        local_obj_markdown_converter = ImagesToMarkdown()
        local_obj_markdown_input = ImagesToMarkdownInput(
            list_images=local_obj_imagesOutput.list_images,
        )
        local_obj_markdown_output = local_obj_markdown_converter.convert(local_obj_markdown_input)

        # Step 3: Generate sections from Markdown
        local_obj_section_generator = SectionGenerator()
        local_obj_section_input = SectionGeneratorInput(str_markdown=local_obj_markdown_output.str_markdown)
        local_obj_section_output = local_obj_section_generator.generate(local_obj_section_input)

        # Step 4: Rewrite sections for clarity and quality
        local_str_overallMarkdown = ""
        for section in local_obj_section_output.list_sections:
            local_obj_rewrite_input = SectionRewriteInput(
                str_nameOftheSection=section.str_title,
                str_markdown=section.str_contents[0]
            )
            local_obj_rewriter = SectionRewriter()
            local_obj_rewrite_output = local_obj_rewriter.rewrite(local_obj_rewrite_input)
            local_str_overallMarkdown += f"\n\n# {local_obj_rewrite_output.str_sectionName}\n\n{local_obj_rewrite_output.str_rewrittenSection}"

        # Save the final Markdown output
        with open(param_obj_input.str_outputPath + "/final_notes.md", "w", encoding="utf-8") as file:
            file.write(local_str_overallMarkdown)