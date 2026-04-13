from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("GRU.pdf")

docs = data.load()

print(docs)
print(docs[0])
print(len(docs))