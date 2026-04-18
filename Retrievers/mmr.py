from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

documents = [
  Document(page_content="Gradient Descent is an optimization algorithm used in Machine Learning"),
  Document(page_content="Gradient Descent minimizes the loss function"),
  Document(page_content="Gradient Descent is an optimization technique that minimized the loss"),
  Document(page_content="Neural networks use gradient descent for training"),
  Document(page_content="Support Vector Machines are supervised learning algorithms"),
]

embedding = HuggingFaceEmbeddings()

vector_store = Chroma.from_documents(documents=documents, embedding=embedding)

similarity = vector_store.as_retriever(
  search_type="similarity",
  search_kwargs={"k": 3}
)

print("\n--------Similarity Search Results--------\n")

similar_docs = similarity.invoke("What is gradient descent?")

for doc in similar_docs:
  print(doc.page_content)
  
mmr = vector_store.as_retriever(
  search_type="mmr",
  search_kwargs={"k": 3}
)

print("\n--------MMR Search Results--------\n")

similar_docs = mmr.invoke("What is gradient descent?")

for doc in similar_docs:
  print(doc.page_content)