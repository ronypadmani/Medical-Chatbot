from dotenv import load_dotenv
import os
import json

from src.helper import (
    filter_to_minimal_docs,
    text_split,
    download_embeddings
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone
from pinecone import ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

TRACK_FILE = "indexed_files.json"
DATA_FOLDER = "data"
INDEX_NAME = "medical-chatbot"

# ---------------------------------------------------
# Create tracking file if not exists
# ---------------------------------------------------

if not os.path.exists(TRACK_FILE):
    with open(TRACK_FILE, "w") as f:
        json.dump([], f)

# ---------------------------------------------------
# Read already indexed PDFs
# ---------------------------------------------------

with open(TRACK_FILE, "r") as f:
    indexed_files = json.load(f)

# ---------------------------------------------------
# Find all PDFs
# ---------------------------------------------------

all_files = [
    f for f in os.listdir(DATA_FOLDER)
    if f.endswith(".pdf")
]

# ---------------------------------------------------
# Find only NEW PDFs
# ---------------------------------------------------

new_files = [
    f for f in all_files
    if f not in indexed_files
]

if len(new_files) == 0:
    print("No new PDF files found.")
    exit()

print("New PDFs Found:")
for file in new_files:
    print(file)

# ---------------------------------------------------
# Load only NEW PDFs
# ---------------------------------------------------

all_docs = []

for pdf in new_files:

    pdf_path = os.path.join(DATA_FOLDER, pdf)

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    for doc in docs:
        doc.metadata["source_file"] = pdf

    all_docs.extend(docs)

# ---------------------------------------------------
# Chunking
# ---------------------------------------------------

minimal_docs = filter_to_minimal_docs(all_docs)

text_chunks = text_split(minimal_docs)

print(f"Total New Chunks: {len(text_chunks)}")

# ---------------------------------------------------
# Embeddings
# ---------------------------------------------------

embeddings = download_embeddings()

# ---------------------------------------------------
# Pinecone Connection
# ---------------------------------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

# ---------------------------------------------------
# Create Index If Not Exists
# ---------------------------------------------------

if not pc.has_index(INDEX_NAME):

    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Pinecone Index Created")

# ---------------------------------------------------
# Connect Existing Index
# ---------------------------------------------------

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)   

# ---------------------------------------------------
# Add ONLY New Chunks
# ---------------------------------------------------

vectorstore.add_documents(text_chunks)

print(f"Added {len(text_chunks)} chunks to Pinecone.")

# ---------------------------------------------------
# Update Indexed File List
# ---------------------------------------------------

indexed_files.extend(new_files)

with open(TRACK_FILE, "w") as f:
    json.dump(indexed_files, f, indent=4)

print("Indexed file list updated.")
print("Done.")