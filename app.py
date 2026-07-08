from flask import Flask, request, jsonify, render_template, redirect, url_for
from src.helper import download_embeddings,filter_to_minimal_docs, text_split
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from src.prompt import *
import json
import os

app =Flask(__name__)

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

INDEXED_FILES = "indexed_files.json"

if not os.path.exists(INDEXED_FILES):
    with open(INDEXED_FILES, "w") as f:
        json.dump([], f)

chat_history = []

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_embeddings()


index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding=embeddings
)

retrieval = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

chatmodel = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.2,
    api_key=os.getenv("MISTRAL_API_KEY")
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)


question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)
rag_chain = create_retrieval_chain(retrieval, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/upload", methods=["GET", "POST"])
@app.route("/upload", methods=["GET", "POST"])
def upload_pdf():

    if request.method == "POST":

        file = request.files.get("file")

        if not file:
            return "No file selected"

        filename = secure_filename(file.filename)

        # Load indexed files list
        with open(INDEXED_FILES, "r") as f:
            indexed_files = json.load(f)

        # Skip if already indexed
        if filename in indexed_files:
            return redirect(url_for("index"))

        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        loader = PyPDFLoader(filepath)

        docs = loader.load()

        minimal_docs = filter_to_minimal_docs(docs)

        chunks = text_split(minimal_docs)

        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

        vectorstore.add_documents(chunks)

        # Save filename to indexed_files.json
        indexed_files.append(filename)

        with open(INDEXED_FILES, "w") as f:
            json.dump(indexed_files, f, indent=4)

        return redirect(url_for("index"))

    return render_template("upload.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form.get("msg")
    input = msg

    response = rag_chain.invoke({
        "input": input,
        "chat_history": chat_history
    })

    chat_history.append(
        HumanMessage(content=input)
    )

    chat_history.append(
        AIMessage(content=response["answer"])
    )

    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080, debug=True)