from langchain import LLMChain, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import AzureChatOpenAI

from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain

from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from langchain.docstore.document import Document

from llm import create_llm
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.document_loaders import Docx2txtLoader

def summarize(path: str, llm, instruction="Write a succint summary of the following text" ) -> str: 
    loader = DirectoryLoader(path, loader_cls=Docx2txtLoader);
    docs = loader.load();
    print( len(docs) )
    
    # Define prompt
    prompt_template = f"""{instruction}
    TEXT:"{{text}}"
    SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text"
    )

    print(stuff_chain.run(docs))
  

llm=create_llm();

path="Sales offerings/Augusta & Co/Project Horizon"
summary_instruction="""We need to find suitable people for a new project. Please rewrite the
    following text so that it highlights what needs to be done and what skills are needed."""

project_summary = summarize(f"./drive/{path}",llm, summary_instruction)

