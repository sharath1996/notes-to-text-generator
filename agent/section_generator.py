import logging
from textwrap import dedent
from typing import List, Union, Optional
from numpy import str_
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
import os
import json

# ---------------------------
# 1. Data Models
# ---------------------------


class Section(BaseModel):

    str_title: str = Field(..., description="The title of the section")
    str_contents: List[str] = Field(..., description="The content of the section, which can include various entries")

class SectionGeneratorInput(BaseModel):

    str_markdown: str = Field(..., description="Markdown representation of the section to be generated")

class SectionGeneratorOutput(BaseModel):
    
    list_sections: List[Section] = Field(default_factory=list, description="List of sections generated from the Markdown input")

# ---------------------------
# 2. SectionGenerator Class
# ---------------------------

class SectionGenerator:
    

    def __init__(self):
        
        self.llm = self._get_llm()
        self._list_sectionNames = []

    def generate(self, param_obj_input: SectionGeneratorInput) -> SectionGeneratorOutput:
        
        local_list_chunks = self._chunk_markdown(param_obj_input.str_markdown)

        local_obj_toc = ToCExtractor(self.llm).extract(local_list_chunks)

        self._list_sectionNames = local_obj_toc.list_sectionTitles

        local_list_sections = self._extract_sections( local_list_chunks)

        return SectionGeneratorOutput(list_sections=local_list_sections)
        
       
    def _get_llm(self):
        
        return AzureChatOpenAI(
        
            azure_deployment=os.environ.get("AZURE_DEPLOYEMENT_GENERIC"),
            max_tokens= 5000
        
        )

    def _chunk_markdown(self, markdown: str) -> List[str]:
        local_obj_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=0
        )
        return local_obj_splitter.split_text(markdown)

    def _extract_sections(self,  param_list_chunks: List[str]) -> list[Section]:
        """
        Extracts sections from the provided markdown based on the current and next section names.
        """

        local_int_chunkIndex = 0
        local_dict_sectionContents = {}

        for local_str_sectionName in self._list_sectionNames:
            local_obj_input = SectionExtractorInput(
                str_markdownChunk=param_list_chunks[local_int_chunkIndex],
                str_nameOfTheCurrentSection=local_str_sectionName,
                str_nameOfTheNextSection=self._get_next_section_name(local_str_sectionName)
            )
            local_obj_extractor = SectionExtractor(self.llm)
            local_obj_output = local_obj_extractor.extract(local_obj_input)
            
            if local_obj_output.bool_isNextSectionPresent:
                local_int_chunkIndex += 1
                local_str_currentSectionContent = local_dict_sectionContents.get(local_str_sectionName, "")
                local_dict_sectionContents[local_str_sectionName] = local_str_currentSectionContent + local_obj_output.str_currentSectionContent 
                local_dict_sectionContents[self._get_next_section_name(local_str_sectionName)] = local_obj_output.str_nextSectionContent
            
            else:
                local_int_chunkIndex += 1
                logging.info(f"Next section {local_str_sectionName} not present in chunk {local_int_chunkIndex}, continuing to next chunk.")
                local_str_currentSectionContent = local_dict_sectionContents.get(local_str_sectionName, "")
                local_dict_sectionContents[local_str_sectionName] = local_str_currentSectionContent + local_obj_output.str_currentSectionContent

        local_list_sections = []
        for local_str_sectionName, local_str_content in local_dict_sectionContents.items():
            
            local_obj_section = Section(
                str_title=local_str_sectionName,
                str_contents=[local_str_content]
            )
        
            local_list_sections.append(local_obj_section)
        
        return local_list_sections
        

    def _get_next_section_name(self, current_section_name: str) -> Optional[str]:
        """
        Returns the next section name based on the current section name.
        If there is no next section, returns None.
        """
        if not self._list_sectionNames:
            return None
        
        try:
            local_int_currentIndex = self._list_sectionNames.index(current_section_name)
            if local_int_currentIndex + 1 < len(self._list_sectionNames):
                return self._list_sectionNames[local_int_currentIndex + 1]
            else:
                return None
        except ValueError:
            return None

## --------------------------- Section Extractor ---------------------------##

class SectionExtractorInput(BaseModel):
    str_nameOfTheCurrentSection: str = Field(..., description="Name of the current section being processed")
    str_nameOfTheNextSection: Optional[str] = Field(None, description="Name of the next section, if available")
    str_markdownChunk: Optional[str] = Field(None, description="Markdown chunk to process, if available")

class SectionExtractorOutput(BaseModel):
    str_currentSectionContent: str = Field(..., description="Content of the section extracted from the markdown")
    bool_isNextSectionPresent: bool = Field(..., description="Indicates if the next section is present in the markdown")
    str_nextSectionContent: Optional[str] = Field(None, description="Content of the next section, if available")


class SectionExtractor:

    def __init__(self, param_obj_llm: AzureChatOpenAI):
        self._obj_llm = param_obj_llm
    
    def extract(self, param_obj_input: SectionExtractorInput) -> SectionExtractorOutput:
        """
        Extracts the section content from the provided markdown based on the current and next section names.
        """
        local_obj_systemPrompt = SystemMessage(dedent(f"""
        You are an expert in splitting the given markdown into sections.
        You will be provided with a markdown document's chunk, current section name and next section name. 
        The markdown chunk is a part of the larger document, and the current section title might be present in the previous chunk and not this chunk.
        So, you should not assume that the current section title is present in the given markdown chunk.
                                                      
        Similarly, the next section might be present in the current chunk, you should be able to identify if the next section is present in the current chunk or not.
        You should split the markdown into parts: 
        1. Content of the current section without any compression, i.e., the content of the current section as it is present in the markdown chunk
        2. Content of the next section if available also indicating if the next section is present or not.
        You should respond with a json object defined by the following schema:
        
        {SectionExtractorOutput.model_json_schema()}
        
        If the next section is not present, set `bool_isNextSectionPresent` to false and `str_nextSectionContent` to null.
        """))
        
        local_obj_humanMessage = HumanMessage(content=f"Extract content for markdown chunk: \n ```{param_obj_input.str_markdownChunk}```")
        local_obj_humanMessage2 = HumanMessage(content=f"Next section name: {param_obj_input.str_nameOfTheNextSection if param_obj_input.str_nameOfTheNextSection else 'None'}")

        local_obj_response = self._obj_llm.with_structured_output(SectionExtractorOutput).invoke(
            [local_obj_systemPrompt, local_obj_humanMessage, local_obj_humanMessage2]
        )
        
        return local_obj_response






## --------------------------- TOC Extractor ---------------------------##


class TOC(BaseModel):
    bool_hasToc: bool = Field(..., description="Indicates if the document has a Table of Contents")
    list_sectionTitles: List[str] = Field(default_factory=list, description="List of section titles in the Table of Contents")


class ToCExtractor:

    def __init__(self, param_obj_llm: AzureChatOpenAI):
        self._obj_llm = param_obj_llm


    def extract(self, param_list_chunks: List[str]) -> TOC:
        """
        Extracts the Table of Contents from the provided list of markdown chunks.
        """
        bool_hasToc = False
        for local_str_chunk in param_list_chunks:
            local_obj_toc = self._extract(local_str_chunk)
            if local_obj_toc.bool_hasToc:
                bool_hasToc = True
                break
        
        if not bool_hasToc:
            return TOC(bool_hasToc=False, list_sectionTitles=[])
        
        return local_obj_toc
    
    def _extract(self, param_str_markdown: str) -> TOC:
        
        local_obj_systemPrompt = SystemMessage(dedent(f"""
        You are an expert in creating a table of contens for a hand-written notes document.
        You will be provided with a markdown document, but the document may not have an explicitly defined header for the Table of Contents.
        You might need to see if there are any paragraphs that appear like table of contents, or list of sections or topics that are said to be defined in the document.
        You should respond with json object defined by following schema:
                                    
        {TOC.model_json_schema()}

        Note that the given markdown may also not have any table of contents, in which case you should set the `bool_hasToc` to false and `list_sectionTitles` to an empty list.
        """))
        local_obj_humanMessage = HumanMessage(content=f"Create topics list if any present in the following text: \n{param_str_markdown}")
        local_obj_response = self._obj_llm.with_structured_output(TOC).invoke(
            [local_obj_systemPrompt, local_obj_humanMessage]
        )
        return local_obj_response

    

    