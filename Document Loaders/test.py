from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

data = TextLoader("notes.txt", encoding="utf-8")
# <langchain_community.document_loaders.text.TextLoader object at 0x00000176A63F0500> Generates this kind of output
# print(data)

splitter = CharacterTextSplitter(chunk_size=5, chunk_overlap=1, separator="")

documents = data.load()

chunks = splitter.split_documents(documents)

print(documents)
print(documents[0].page_content)
"""
Produces an output in the form of a list
[Document(metadata={'source': 'notes.txt'}, page_content='Hello how are you \n\nI want to see what can I do and also I need your help \nplease help me')]
"""

print(len(chunks))
print(chunks)

for i in chunks:
  print(i.page_content)
  print()