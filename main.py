from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# data = TextLoader("Document Loaders/notes.txt")
data = PyPDFLoader("Document Loaders/deeplearning.pdf")
documents = data.load()

splitter = RecursiveCharacterTextSplitter(
  chunk_size = 1000,
  chunk_overlap = 200
)

chunks = splitter.split_documents(documents)

template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are an intelligent AI that summarizes the text"), 
    ("human", "{data}")
  ]
)

model = ChatMistralAI(model="mistral-small-2506")

prompt = template.format_messages(data = documents[0].page_content)

response = model.invoke(prompt)

print(response.content)