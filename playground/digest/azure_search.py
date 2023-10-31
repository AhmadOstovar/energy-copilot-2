
import os
from dotenv import load_dotenv
from langchain.vectorstores.azuresearch import AzureSearch

from embeddings import create_embeddings

def create_azure_search() -> AzureSearch:
    load_dotenv()

    AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
    AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME")
    AZURE_SEARCH_ADMIN_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")

    print( f"Connecting to {AZURE_SEARCH_INDEX_NAME} @ {AZURE_SEARCH_SERVICE_ENDPOINT}")

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        azure_search_key=AZURE_SEARCH_ADMIN_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        embedding_function=create_embeddings().embed_query,
    )

    return vector_store
