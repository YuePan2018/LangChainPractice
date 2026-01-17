from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import os

loader = PyPDFLoader("nke-10k-2023.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

vector_store = InMemoryVectorStore(embeddings)
part_splits = all_splits[:15]
ids = vector_store.add_documents(documents=part_splits)
results = vector_store.similarity_search("When is NIKE founded", k=2)
print(results[0].page_content)