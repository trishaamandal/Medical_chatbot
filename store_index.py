# from dotenv import load_dotenv
# import os
# from src.helper import (
#     load_pdf_file,
#     filter_to_minimal_docs,
#     text_split,
#     download_hugging_face_embeddings,
# )
# from pinecone import Pinecone
# from pinecone import ServerlessSpec
# from langchain_pinecone import PineconeVectorStore

# load_dotenv()


# PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# extracted_data = load_pdf_file(data="data/")
# filter_data = filter_to_minimal_docs(extracted_data)
# text_chunks = text_split(filter_data)

# embeddings = download_hugging_face_embeddings()

# pinecone_api_key = PINECONE_API_KEY
# pc = Pinecone(api_key=pinecone_api_key)


# index_name = "medical-chatbot"  # change if desired

# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )

# index = pc.Index(index_name)


# docsearch = PineconeVectorStore.from_documents(
#     documents=text_chunks,
#     index_name=index_name,
#     embedding=embeddings,
# )


import os
import time
from dotenv import load_dotenv

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]


# Load and process PDFs
extracted_data = load_pdf_file(data="data/")
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)


# Embeddings
embeddings = download_hugging_face_embeddings()


# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.has_index(index_name):
        time.sleep(2)

index = pc.Index(index_name)

# OPTIONAL during development
# index.delete(delete_all=True)

# Store documents
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
    namespace="medical-pdfs",
)
