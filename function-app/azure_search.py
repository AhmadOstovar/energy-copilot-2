
import os
from langchain.vectorstores.azuresearch import AzureSearch
from embeddings import create_embeddings
from azure.search.documents.indexes.models import (
    SemanticSettings,
    SemanticConfiguration,
    PrioritizedFields,
    SemanticField
)

def create_azure_search() -> AzureSearch:

    AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
    AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME")
    AZURE_SEARCH_ADMIN_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")

    print( f"Connecting to {AZURE_SEARCH_INDEX_NAME} @ {AZURE_SEARCH_SERVICE_ENDPOINT}")

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        azure_search_key=AZURE_SEARCH_ADMIN_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        embedding_function=create_embeddings().embed_query,
        semantic_configuration_name='config',
            semantic_settings=SemanticSettings(
                default_configuration='config',
                configurations=[
                    SemanticConfiguration(
                        name='config',
                        prioritized_fields=PrioritizedFields(
                            title_field=SemanticField(field_name='content'),
                            prioritized_content_fields=[SemanticField(field_name='content')],
                            prioritized_keywords_fields=[SemanticField(field_name='metadata')]
                        ))
                ])
        )

    return vector_store
