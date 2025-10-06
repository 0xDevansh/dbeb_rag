import os
import streamlit as st
import asyncio
import uuid
from typing import Dict, Any, List
import nest_asyncio

# Apply nest_asyncio at the very top
nest_asyncio.apply()

# --- Environment Variables ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyCajA-t68BWLwRSrc3qolSBNRGw70xMnzo"
os.environ["HF_HUB_ENABLE_SYMLINKS"] = "1"

# --- Imports ---
import torch
import logging
from functools import partial

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessageChunk, SystemMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

# Reduce logging spam
logging.getLogger('google.api_core').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)

# --- Hard-coded System Prompt Constant ---
SYSTEM_PROMPT = """
You are a helpful and professional assistant named 'BEBot'. You help members of the Department of Biochemistry Biotechnology at IIT Delhi with their queries about information stored in archives and meeting notes.
You are an expert at retrieving information from a knowledge base.
Your goal is to answer user questions based *only* on the context provided by the document retriever tool.
If the information is not in the provided documents, you must state that you cannot answer the question with the available information.
Do not make up answers. Be concise and clear in your responses.
IMPORTANT: Prioritise recent documents over older ones when answering questions.
At the end of your response, always include the sources used in the FINAL response as a hyperlink by appending the file name (source field in metadata) in https://beb.iitd.ac.in/
If the source contains /pdf/, remove it. Only keep the file name.
Example: [(1)](https://beb.iitd.ac.in/DFB-Minutes-2019-2020-1.pdf)
"""

# --- Retriever Initialization ---
@st.cache_resource
def initialize_retriever():
    """Initializes the retriever and required models."""
    print("Initializing models and retriever connection...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ Using {device} for embeddings.")

    model_kwargs = {'device': device}
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs
    )
    
    url = st.secrets.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url, prefer_grpc=False, api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.JHnCoW2lyO0LMWxlvoRTNaLu-ioFTtYYp3x1EyKBP_0")
    
    qdrant = QdrantVectorStore(
        client=client,
        collection_name="dbeb",
        embedding=embeddings,
    )
    
    print("✅ Models and retriever connection initialized.")
    return qdrant.as_retriever()

# --- LangGraph State ---
class State(MessagesState):
    pass

# --- LangGraph Nodes ---
async def agent_node(state: State, llm_with_tools) -> Dict[str, Any]:
    """Invokes the agent LLM."""
    print("NODE: AGENT")
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}

# --- Main App Logic ---
st.set_page_config(page_title="🤖 Agentic RAG Chatbot", layout="wide")
st.title("🤖 Agentic RAG Chatbot")

retriever = initialize_retriever()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- NEW: Custom function to stream response and display sources ---
async def get_response_and_display_sources(graph, user_input, config):
    """
    Streams the response and displays retrieved documents in an expander.
    This function replaces st.write_stream.
    """
    full_response = ""
    # Use st.chat_message to create a new chat bubble
    with st.chat_message("assistant"):
        # Placeholder for the documents expander
        expander_placeholder = st.empty()
        # Placeholder for the streaming response
        response_placeholder = st.empty()

        # Create the input for the graph
        system_message = SystemMessage(content=SYSTEM_PROMPT)
        human_message = HumanMessage(content=user_input)
        graph_input = {"messages": [system_message, human_message]}

        # Stream events from the graph
        async for event in graph.astream_events(graph_input, config, version="v2"):
            kind = event["event"]

            # This event fires when the retriever tool finishes its work
            # if kind == "on_tool_end":
            #     documents: List[Document] = event["data"].get("output")
            #     if documents:
            #         for i, doc in enumerate(documents):
            #             print(f"**Chunk {i+1}:**")
            #             print(doc)

            # This event fires for each chunk of the LLM's response
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
        
        # Display the final response without the typing cursor
        response_placeholder.markdown(full_response)
        
    return full_response

# Handle user input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Define tools and LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, streaming=True)
    
    retriever_tool = Tool(
        name="document_retriever",
        description="Searches and returns relevant information from a knowledge base of documents. Use this for any user question that requires information from the documents.",
        coroutine=retriever.ainvoke,
        func=retriever.invoke,
    )
    tools = [retriever_tool]
    llm_with_tools = llm.bind_tools(tools)
    
    # Create the graph
    agent_node_with_llm = partial(agent_node, llm_with_tools=llm_with_tools)
    tool_node = ToolNode(tools=tools)

    memory = InMemorySaver()
    workflow = StateGraph(State)
    workflow.add_node("agent", agent_node_with_llm)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    graph = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # MODIFICATION: Call the new custom function instead of st.write_stream
    full_response = asyncio.run(get_response_and_display_sources(graph, prompt, config))
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})