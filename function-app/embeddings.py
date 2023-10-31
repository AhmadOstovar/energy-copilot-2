import os
from langchain.embeddings.openai import OpenAIEmbeddings

def create_embeddings() -> OpenAIEmbeddings:

    AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION")
    AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION")
    OPENAI_API_EMBEDDING_VERSION = os.environ.get("OPENAI_API_EMBEDDING_VERSION")
    OPENAI_API_EMBEDDING_DEPLOYMENT_NAME = os.environ.get("OPENAI_API_EMBEDDING_DEPLOYMENT_NAME")
    
    return OpenAIEmbeddings(
            model=OPENAI_API_EMBEDDING_VERSION,
            deployment=OPENAI_API_EMBEDDING_DEPLOYMENT_NAME,
            openai_api_key=AZURE_OPENAI_KEY,
            openai_api_base=AZURE_OPENAI_ENDPOINT,
            openai_api_type='azure',
            openai_api_version=OPENAI_API_VERSION,
            chunk_size=1
        )