import getpass
import os
from groq import Groq
import bs4
import dotenv
import streamlit as st
import tempfile
from langchain_core.messages import AIMessage, HumanMessage

dotenv.load_dotenv()

# if not os.environ.get("GOOGLE_API_KEY"):
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Gemini: ")

# Usign groq
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"


from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

@st.cache_resource
def get_llm():
    return init_chat_model("gemini-2.5-flash", model_provider="google_genai", transport="grpc")


@st.cache_resource
def get_Groqllm():
    return ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", transport="grpc")

st.title("PDF-powered Q&A Chatbot")

llm = get_Groqllm()
embeddings = get_embeddings()

# prompt = hub.pull("rlm/rag-prompt")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

Question: {question}

Context: {context}

Answer:"""
)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = st.session_state.vectorStore.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorStore" not in st.session_state:
    st.session_state.vectorStore = None
if "graph" not in st.session_state:
    st.session_state.graph = None

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if st.session_state.graph is None:
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    st.session_state.graph = graph_builder.compile()

if uploaded_file and st.session_state.vectorStore is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.spinner("Processing PDF..."):
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        all_splits = text_splitter.split_documents(docs)

        st.session_state.vectorStore = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings,
        )

    os.unlink(tmp_file_path)

    st.success("PDF processed successfully!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.vectorStore:
    if prompt := st.chat_input("Ask a question about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("Thinking..."):
                for chunk in st.session_state.graph.stream({"question": prompt}):
                    if "answer" in chunk.get("generate", {}):
                        full_response += chunk["generate"]["answer"]
                        message_placeholder.markdown(full_response + " ")
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload a PDF to begin.")