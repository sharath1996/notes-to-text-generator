from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from textwrap import dedent
from langchain_community.retrievers.wikipedia import WikipediaRetriever
import os

class ResourceFetcherInput(BaseModel):
    """
    Input model for the content analysis process.
    """
    str_content: str = Field(..., description="Content to be analyzed")


class Resource(BaseModel):
    str_topic: str = Field(..., description="Topic of the resource")
    str_contents:str = Field(..., description="Contents of the resource")

class Topics(BaseModel):
    """
    Model to hold a list of resources.
    """
    list_resources: list[str] = Field(..., description="List of the topics that are described in the content, limit the topics to maximum of 10 topics")


class ResourceFetcherOutput(BaseModel):
    """
    Output model for the content analysis process.
    """
    list_resources: list[Resource] = Field(..., description="Various resources related to the content")

class ResourceFetcher:

    def __init__(self):
        self._obj_llm = AzureChatOpenAI(
            azure_deployment=os.environ.get("DEPLOYEMENT_REASONING"),
        )

    def fetch_resources(self, param_obj_input: ResourceFetcherInput) -> ResourceFetcherOutput:
        """
        Fetch resources based on the provided content.
        
        :param param_obj_input: Input data containing content to analyze.
        :return: Output data containing resources related to the content.
        """
        local_obj_topics = TopicsFetcher().fetch_topics(param_obj_input)
        local_list_resources = []
        for local_str_topic in local_obj_topics.list_resources:
            local_str_wikipedia = WikiPediaFetcher().fetch_wikipedia(local_str_topic)
            local_obj_rewriterInput = TopicRewriterInput(
                str_topic=local_str_topic,
                str_handwriitenContent=param_obj_input.str_content,
                str_otherContents=local_str_wikipedia
            )

            local_str_rewrittenTopic = TopicRewriter().run(local_obj_rewriterInput)
            local_list_resources.append(Resource(
                str_topic=local_str_topic,
                str_contents=local_str_rewrittenTopic
            ))


        local_obj_output = ResourceFetcherOutput(list_resources=local_list_resources)

        return local_obj_output
        


class TopicsFetcher:

    def __init__(self):
        self._obj_llm = AzureChatOpenAI(
            azure_deployment=os.environ.get("DEPLOYEMENT_REASONING"),
        )

    def fetch_topics(self, param_obj_input: ResourceFetcherInput) -> Topics:
        """
        Fetch topics based on the provided content.
        
        :param param_obj_input: Input data containing content to analyze.
        :return: Output data containing topics related to the content.
        """
        ...

        local_obj_systemPrompt = SystemMessage(dedent(f"""
        You are an expert in extracting topics from a given handwritten notes. 
        You task is to analyze the content and list the topics that are described in the content.
        These topics will further be used to search for relevant resources, so the topics should be specific and relevant along with the keywords.
        The topics should be listed in a JSON format with the following structure:
        {Topics.model_json_schema()})"""))
        
        local_obj_humanPrompt = HumanMessage(dedent(f"""
        Please analyze the following content and extract the topics:
        {param_obj_input.str_content}
        """))
        local_obj_messages = [local_obj_systemPrompt, local_obj_humanPrompt]
        local_obj_response = self._obj_llm.with_structured_output(Topics).invoke(local_obj_messages)

        return local_obj_response
        

class TopicRewriterInput(BaseModel):
    """
    Input model for the topic rewriting process.
    """
    str_topic: str = Field(..., description="Topic to be rewritten")
    str_handwriitenContent: str = Field(..., description="Content to analyze for rewriting the topic")
    str_otherContents: str = Field(..., description="Other contents to analyze for rewriting the topic")


class TopicRewriter:

    def __init__(self):
        self._obj_llm = AzureChatOpenAI(
            azure_deployment=os.environ.get("DEPLOYEMENT_REASONING"),
        )
    
    def run(self, param_obj_input:TopicRewriterInput) -> str:
        """
        Rewrite the topic based on the provided content.
        
        :param param_str_topic: Topic to be rewritten.
        :param param_str_content: Content to analyze for rewriting the topic.
        :return: Rewritten topic.
        """
        
        local_obj_systemPrompt = SystemMessage(dedent("""
        You are expert in writing technical textbooks which are used for teaching Master's and PhD students.
        Your task is to write a detailed text based on the topic given to you, handwritten notes from the faculty and relevant additional resources given to you.
        The text should be detailed, including all the relevant explanations, formulas and refereces.
        All the equations should be written in latex format.
        """))
        
        local_obj_humanPrompt = HumanMessage(dedent(f"""
        Please analyze the following content and rewrite the topic:
        Topic: {param_obj_input.str_topic}
        Content: {param_obj_input.str_handwriitenContent}

        Additional resources:
        {param_obj_input.str_otherContents}
        You should only return the re-written topic.
        Remember you are not a conversational agent, so you should not return any greetings or salutations.
        You should only return the re-written topic in detailed format.
        """))
        
        local_obj_messages = [local_obj_systemPrompt, local_obj_humanPrompt]
        local_obj_response = self._obj_llm.invoke(local_obj_messages)

        return local_obj_response.content
    
class WikiPediaFetcher:

    def __init__(self):
        self._obj_llm = AzureChatOpenAI(
            azure_deployment=os.environ.get("DEPLOYEMENT_REASONING"),
        )

    def fetch_wikipedia(self, param_str_topic:str) -> str:
        """
        Fetch Wikipedia resources based on the provided content.
        
        :param param_obj_input: Input data containing content to analyze.
        :return: Output data containing Wikipedia resources related to the content.
        """
        
        local_obj_retriever = WikipediaRetriever()
        local_list_docs = local_obj_retriever.invoke(param_str_topic)

        local_str_docs = "\n".join([doc.page_content for doc in local_list_docs])
        return local_str_docs
        

class ArxivFetcher:

    def __init__(self):
        self._obj_llm = AzureChatOpenAI(
            azure_deployment=os.environ.get("DEPLOYEMENT_REASONING"),
        )

    def fetch_arxiv(self, param_str_topic:str) -> str:
        """
        Fetch arXiv resources based on the provided content.
        
        :param param_obj_input: Input data containing content to analyze.
        :return: Output data containing arXiv resources related to the content.
        """
        
        ...