from io import StringIO
import os
import re
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import pandas as pd
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
from tabula import read_pdf 
from tabula import convert_into
from tabulate import tabulate 
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.pdf import UnstructuredPDFLoader
from PyPDF2 import PdfReader
from langchain.document_loaders.blob_loaders import Blob

class PDFLoader(PyPDFLoader):
    def __init__(
        self,
        file_path: str,
        password: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict] = None,
        extract_images: bool = False,
        table_chunk_size: Optional[int] = None
    ) -> None:
        self.table_chunk_size = table_chunk_size
        super().__init__(file_path, password, headers, extract_images)

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        yield from self.parse_document_and_tables(self.file_path)  

    def parse_document_and_tables(self, file_path:str) -> List[Document]:
        blob = Blob.from_path(self.file_path)
        pages = self.parser.parse(blob)

        documents = []

        prev_table = None
        for index, page in enumerate(pages):
            print( f"Processing page {index+1}/{len(pages)}")

            documents.append( page )

            tables = read_pdf(file_path, stream=True, pages=[index+1], pandas_options={'header': None})

            page_lines = page.page_content.split('\n')
            page_first_row = re.sub(r'[^a-zA-Z]', '', page_lines[0]) 
            page_last_row = re.sub(r'[^a-zA-Z]', '', page_lines[-1]) 

            for t, table in enumerate(tables):
                table.dropna(how='all', axis=1, inplace=True) 

                csv = table.to_csv( sep='\t', index=False, header=False)
                table_lines = csv.split('\n')
                if table_lines[-1] == '': 
                    table_lines = table_lines[:-1]

                table_first_row = re.sub(r'[^a-zA-Z]', '', table_lines[0]) 
                table_last_row = re.sub(r'[^a-zA-Z]', '', table_lines[-1]) 

                # If there was a table on the previous page 
                # that might continue on the current page
                if not prev_table is None: 
                    # if the current page starts with the table and 
                    # the table has the same column count as the previous table
                    if len(prev_table.columns) == len(table.columns) and t==0 and page_first_row.startswith(table_first_row):
                        prev_table.columns = table.columns
                        table = pd.concat( [prev_table,table],axis=0,ignore_index=True)
                    else:
                        documents.extend(self.create_document(prev_table,page.metadata))

                # If the table is the last table on the page, and it 
                # ends with the same row as the page, keep it in case 
                # it continues on the next page
                if t == len(tables)-1 and page_last_row.endswith(table_last_row): 
                    prev_table = table
                else:
                    prev_table = None
                    documents.extend(self.create_document(table,page.metadata))
                
        if not prev_table is None:
            documents.extend(self.create_document(prev_table,page.metadata))

        return documents

    """
    Creates documents for the given dataframe, splitting it into smaller
    sizes if the text representation exceeds the given chunk size.
    """
    def create_document(self, table: pd.DataFrame, metadata:dict) -> List[Document]:
        # Make the first row of the table be the table headers
        table.columns = table.iloc[0]
        table = table.iloc[1:]
        table.reset_index(drop=True)


        # Convert table to csv to see how many parts to split it into
        full_table_as_csv = table.to_csv( sep='\t', index=False)


        if len(full_table_as_csv) > self.table_chunk_size:
            no_chunks = len(full_table_as_csv) // self.table_chunk_size
            no_rows_per_chunk = len(table) // no_chunks

            chunks = [table.iloc[i:i + no_rows_per_chunk] for i in range(0, len(table), no_rows_per_chunk)]
        else: 
            chunks = [table]

        documents: List[Document] = []
        for chunk in chunks:
            chunk_csv = chunk.to_csv( sep='\t', index=False)
            documents.append(Document(page_content=chunk_csv,metadata=metadata))

        return documents


file_path = "C:/Users/A550191/git/energy-copilot/playground/digest/drive/Food/Food Calories List.pdf"
loader = PDFLoader(file_path,table_chunk_size=400)
docs = loader.load_and_split()


