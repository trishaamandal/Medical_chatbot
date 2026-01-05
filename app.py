from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# --------------------
# App & Environment
# --------------------
app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]


# --------------------
# Embeddings & Vector Store
# --------------------
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace="medical-pdfs",  # MUST match ingestion
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# --------------------
# LLM (Groq)
# --------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0
)


# --------------------
# Prompt & RAG Chain
# --------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# --------------------
# Routes
# --------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)

    response = rag_chain.invoke({"input": msg})
    answer = response["answer"]

    print("Response:", answer)
    return answer


# --------------------
# Run
# --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
