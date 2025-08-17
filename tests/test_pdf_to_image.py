from agent.pages_to_images import PagesToImages, PagesToImagesInput
from agent.images_to_markdown import ImagesToMarkdown, ImagesToMarkdownInput
from agent.section_generator import SectionGenerator, SectionGeneratorInput
from agent.re_writer import SectionRewriteInput, SectionRewriter

import json

def test_pdf_to_images_convert():

    local_obj_input = PagesToImagesInput(str_inputPath="GenAI-IITM.pdf", str_outputPath=".images")
    local_obj_converter = PagesToImages()
    local_obj_output = local_obj_converter.convert(local_obj_input)
    with open(".data/data_pages_to_images.json", "w") as local_obj_file:
        json.dump(local_obj_output.model_dump(), local_obj_file, indent=4)

def test_images_to_markdown_convert():
    with open(".data/data_pages_to_images.json", "r") as local_obj_file:
        local_obj_input = json.load(local_obj_file)
    
    local_obj_input = ImagesToMarkdownInput(**local_obj_input)
    local_obj_converter = ImagesToMarkdown()
    local_obj_output = local_obj_converter.convert(local_obj_input)
    
    with open(".data/data_images_to_markdown.json", "w") as local_obj_file:
        json.dump(local_obj_output.model_dump(), local_obj_file, indent=4)
    
    with open(".data/data_images_to_markdown.md", "w", encoding="utf-8") as local_obj_file:
        local_obj_file.write(local_obj_output.str_markdown)

def test_section_generator_generate():
    with open(".data/data_images_to_markdown.md", "r", encoding="utf-8") as local_obj_file:
        local_str_markdown = local_obj_file.read()
    
    local_obj_input = SectionGeneratorInput(str_markdown=local_str_markdown)
    local_obj_generator = SectionGenerator()
    local_obj_output = local_obj_generator.generate(local_obj_input)
    
    with open(".data/data_section_generator.json", "w") as local_obj_file:
        json.dump(local_obj_output.model_dump(), local_obj_file, indent=4)
    

def test_section_rewriter_rewrite():
    with open(".data/data_section_generator.json", "r") as local_obj_file:
        local_list_input = json.load(local_obj_file)
    
    local_str_overallMarkdown = ""

    for local_dict_input in local_list_input["list_sections"]:
        local_obj_input = SectionRewriteInput(str_nameOftheSection=local_dict_input["str_title"], str_markdown=local_dict_input["str_contents"][0])
        local_obj_rewriter = SectionRewriter()
        local_obj_output = local_obj_rewriter.rewrite(local_obj_input)
        local_str_overallMarkdown += f"\n\n# {local_obj_output.str_sectionName}\n\n{local_obj_output.str_rewrittenSection}"
        
        with open(".data/section.json", "w") as local_obj_file:
            json.dump(local_obj_output.model_dump(), local_obj_file, indent=4)
    
    with open(".data/data_section_rewriter.md", "w", encoding="utf-8") as local_obj_file:
        local_obj_file.write(local_str_overallMarkdown)


def test_raw_section_conversion():

    with open(".data/data_section_generator.json", "r") as local_obj_file:
        local_list_input = json.load(local_obj_file)
    
    local_str_overallMarkdown = ""

    for local_dict_input in local_list_input["list_sections"]:
        with open(f".data/{local_dict_input["str_title"].replace(' ', '_')}.md", "w", encoding="utf-8") as local_obj_file:
            local_obj_file.write(local_dict_input["str_contents"][0])