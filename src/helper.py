from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


# Extract Data form the PDF file
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    
    documents=loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filter a list of documents to only include the page content and metadata.
    """ 
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


# Split the text into chunks

def text_split(minimal_docs):
    text_spitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50,)
    texts_chunk = text_spitter.split_documents(minimal_docs)

    filtered_chunks = [
        chunk
        for chunk in texts_chunk
        if len(chunk.page_content.strip()) > 200
    ]

    return filtered_chunks


# Download the embeddings for HuggingFace

def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )

    return embeddings