from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

url = "https://www.apple.com/in/macbook-pro/"

data = WebBaseLoader(url)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

document = data.load()

print(document)
print(len(document))
print(document[0].page_content)

print("-----------------------------------------")

chunks = splitter.split_documents(document)
for i in chunks:
  print(i.page_content)
  print("----------------------")
  
print(len(chunks))