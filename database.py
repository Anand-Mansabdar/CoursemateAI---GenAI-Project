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
from langchain_mistralai import MistralAIEmbeddings
embedding_model = MistralAIEmbeddings(model="mistral-embed")
vector_store = Chroma.from_documents(
  documents=documents,
  embedding=embedding_model,
  persist_directory="chroma_db"
)

# Store in Chroma
from langchain_community.vectorstores import Chroma