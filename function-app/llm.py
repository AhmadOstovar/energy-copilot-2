import logging
import os
import openai
from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI

def create_llm() -> AzureChatOpenAI:
    AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION")
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get(
        "AZURE_OPENAI_CHATGPT_DEPLOYMENT")

    openai.api_key = AZURE_OPENAI_KEY
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_type = 'azure'
    openai.api_version = OPENAI_API_VERSION  

    llm = AzureChatOpenAI(
        openai_api_base=AZURE_OPENAI_ENDPOINT,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_type='azure',
        deployment_name=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        openai_api_key=AZURE_OPENAI_KEY,
        model="gpt-4-32k",
        temperature=0.3
    )

    return llm
