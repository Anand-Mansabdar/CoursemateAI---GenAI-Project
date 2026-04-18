from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
  load_max_docs=2, # Number of papers to retrieve
  load_all_available_meta=True
)

docs = retriever.invoke("Large Language Model")

for i, doc in enumerate(docs):
  print(f"\nResult {i+1}")
  print("Title:", doc.metadata.get("Title"))
  print("Authors:", doc.metadata.get("Authors"))
  print("Summary:", doc.page_content[:500]) # First 500 characters