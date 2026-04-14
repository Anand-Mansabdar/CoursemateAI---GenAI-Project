from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

data = PyPDFLoader("GRU.pdf")

splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)

docs = data.load()

chunks = splitter.split_documents(docs)

print(docs)
print(docs[0])
print(len(docs))
print("-------------------------------------")

for i in chunks:
  print(i.page_content)

print(len(chunks))