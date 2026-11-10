import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from .config import DOCS_GLOB, DOCS_DIR

def load_documents(docs_path=DOCS_DIR):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The path \"{docs_path}\" does not exist.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob=DOCS_GLOB,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"Not .txt files found inside \"{docs_path}\"")
    
    return documents