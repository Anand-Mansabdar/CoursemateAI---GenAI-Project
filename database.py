# Data Ingestion File to create the database just once
from dotenv import load_dotenv
load_dotenv()

# Load PDF
from langchain_community.document_loaders import PyPDFLoader
data = PyPDFLoader("Document Loaders/deeplearning.pdf")
documents = data.load()

# Split into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
  chunk_size = 1000,
  chunk_overlap = 200
)

chunks = splitter.split_documents(documents)

# Create Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# Store in Chroma
from langchain_community.vectorstores import Chroma
vector_store = Chroma.from_documents(
  documents=documents,
  embedding=embedding_model,
  persist_directory="chroma_db"
)
