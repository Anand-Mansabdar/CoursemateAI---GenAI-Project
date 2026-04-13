from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/macbook-pro/"

data = WebBaseLoader(url)

document = data.load()

print(document)
print(len(document))
print(document[0].page_content)