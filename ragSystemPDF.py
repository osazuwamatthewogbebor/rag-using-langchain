import getpass
import os
import bs4
import dotenv

dotenv.load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Gemini: ")

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorStore = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)


# 1. Indexing pipeline

    # Load data using document loaders
loader = PyPDFLoader("./fyResume.pdf")

docs = loader.load()

    # Split documents into smaller chunks using text splitters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100 )
all_splits = text_splitter.split_documents(docs)

    # Store store and index splits using vectorstore or embeddings model
_ = vectorStore.add_documents(documents=all_splits)
prompt = hub.pull("rlm/rag-prompt")

# 2. Retrieval and Generation

    # Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

    # Retrieve relevant splits based on user's query using a Retriever
def retrieve(state: State):
    retrieved_docs = vectorStore.similarity_search(state["question"])
    return {"context": retrieved_docs}

    # Generate answer using the query and retrieved chunk as prompt
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "What skills are on the resume?"})
print(response["answer"])
 
# Tool LangGraph

