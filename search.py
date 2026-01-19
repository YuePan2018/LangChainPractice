from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import os
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi

api_key = os.getenv("DASHSCOPE_API_KEY")
# load and split the document
loader = PyPDFLoader("nke-10k-2023.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
# create vector store
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=api_key,
    )
vector_store = InMemoryVectorStore(embeddings)
part_splits = all_splits[:15]
ids = vector_store.add_documents(documents=part_splits)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# create an agent in with Tongyi
model = ChatTongyi(model_name="qwen-turbo-2025-04-28", dashscope_api_key=api_key)
agent = create_agent(
    model=model,
    tools=[retrieve_context],
    system_prompt=(
        "You have access to a tool that retrieves context. "
        "You must use the tool to help answer user queries."
        )
)
output = agent.invoke(
        {"messages": [{"role": "user", "content": "When is Nike Founded?"}]}
    )
print(output)