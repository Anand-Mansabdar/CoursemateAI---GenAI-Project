from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

docs = [
  Document(page_content="Gradient Descent is an optimization algorithm used in Machine Learning"),
  Document(page_content="Gradient Descent minimizes the loss function"),
  Document(page_content="Gradient Descent is an optimization technique that minimized the loss"),
  Document(page_content="Neural networks use gradient descent for training"),
  Document(page_content="Support Vector Machines are supervised learning algorithms"),
]

embedding = HuggingFaceEmbeddings()

vector_store = Chroma.from_documents(documents=docs, embedding=embedding)

retriever = vector_store.as_retriever()

llm = ChatMistralAI(model="mistral-small-latest")

multi_query_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

query = "What is Gradient Descent?"

docs = multi_query_retriever.invoke(query)

for d in docs:
  print(d.page_content)