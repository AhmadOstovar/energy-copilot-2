import logging
from typing import List, Tuple
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from azure_search import create_azure_search
from embeddings import create_embeddings
from langchain.docstore.document import Document
from langchain.retrievers import AzureCognitiveSearchRetriever

from llm import create_llm

def create_vector_store():
    return create_azure_search()
#    return FAISS.load_local("faiss_index", create_embeddings())

def print_docs(docs:List[Document]):
    for doc in docs:
        print(doc.page_content)

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

db = create_vector_store()
llm = create_llm()
langchain.verbose = True

query="What was the purpose of the red wolf project?"

retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 50, 'score_threshold':0.8}
)

qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever,chain_type="stuff",verbose=True)
result = qa_chain({"query": query})
print(result)