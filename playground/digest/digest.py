import os

from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

from langchain.docstore.document import Document

from pathlib import Path
from typing import List
from azure_search import create_azure_search
from embeddings import create_embeddings
from excel_loader import ExcelLoader
from llm import create_llm
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders.xml import UnstructuredXMLLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PDFMinerLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores.azuresearch import AzureSearch

from playground.digest.pdf_loader import PDFLoader

def createPdfLoader(path):
    return PDFLoader(path,table_chunk_size=400)

def createXmlLoader(path):
    return UnstructuredXMLLoader(path);

def createCsvLoader(path):
    return CSVLoader(path);

def createTxtLoader(path):
    return TextLoader(path);

def createXlsLoader(path):
    return ExcelLoader(path);

def createPptLoader(path):
    return UnstructuredPowerPointLoader(path);

def createDocLoader(path):
    return Docx2txtLoader(path);

# Define a dictionary to map file extensions to their respective loaders
loaders = {
    '.pdf': createPdfLoader,
    '.xml': createXmlLoader,
    '.csv': createCsvLoader,
    '.txt': createTxtLoader,
    '.xlsx': createXlsLoader,
    '.pptx': createPptLoader,
    '.docx': createDocLoader
}

def digest_file(path:Path) -> List[Document]:
    # Given a file path, extract the file extension
    file_extension = os.path.splitext(path)[1]

    # Use the dictionary to execute the appropriate function
    if file_extension in loaders:
        return loaders[file_extension](path.as_posix()).load()
    else:
        print(f"No handler found for {file_extension} files.")
    
    return []

def digest(*paths: str ) -> List[Document]: 
    all_docs = []
    for path in paths:
        p = Path(path)
        items = list(p.rglob('**/*'))

        for i in items:
            if i.is_file():
                docs = digest_file(i)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
                splits = text_splitter.split_documents(docs)

                all_docs.extend(splits)

    return all_docs

def createFAISS():
    print( "Creating db")

    docs = digest(f"./drive/CVs")
    db = FAISS.from_documents(docs,create_embeddings())
    db.save_local("faiss_index")

def testFAISS():
    db = FAISS.load_local("faiss_index")

    query = "Which persons are fluent in English?"
    docs = db.similarity_search(query)
    print(docs)

def createAzureCognitiveSearch():
    print( "Connecting to index")
    vector_store: AzureSearch = create_azure_search()
    print( "Digesting documents")
    docs = digest(f"./playground/digest/drive/Sales offerings/ABO Wind/")
                               
    print( f"Adding {len(docs)} documents")

    index=1
    for doc in docs:
        print( f"adding doc {index}/{len(docs)}: {doc.page_content}");
        vector_store.add_documents(documents=[doc])
        index+=1

    print( "Done.")

def testAzureCognitiveSearch():
    vector_store: AzureSearch = create_azure_search()

    docs = vector_store.similarity_search(
        query="What is the purpose of project red wolf?",
        k=20,
        search_type="hybrid"
    )
    for doc in docs:
        print(doc.page_content)

createAzureCognitiveSearch()
#testAzureCognitiveSearch()