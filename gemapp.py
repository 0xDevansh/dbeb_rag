import os
import streamlit as st
import asyncio
import uuid
from typing import Dict, Any

import nest_asyncio
nest_asyncio.apply()

os.environ["GOOGLE_API_KEY"] = "AIzaSyCajA-t68BWLwRSrc3qolSBNRGw70xMnzo"
os.environ["HF_HUB_ENABLE_SYMLINKS"] = "1"

import torch
import logging
logging.getLogger('google.api_core').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_google_genai import ChatGoogleGenerativeAI

@st.cache_resource
def initialize_retriever():
    print("Initializing models and retriever connection...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ Using {device} for embeddings.")

    model_kwargs = {'device': device}
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs
    )
    
    url = "http://localhost:6333"
    client = QdrantClient(url=url, prefer_grpc=False)
    
    qdrant = QdrantVectorStore(
        client=client,
        collection_name="dbeb",
        embedding=embeddings,
    )
    
    print("✅ Models and retriever connection initialized.")
    return qdrant.as_retriever()

class State(MessagesState):
    pass

async def agent_node(state: State, llm_with_tools) -> Dict[str, Any]:
    print("NODE: AGENT")
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}

if 'event_loop' not in st.session_state:
    st.session_state.event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(st.session_state.event_loop)

st.set_page_config(page_title="🤖 Agentic RAG Chatbot", layout="wide")
st.title("🤖 Agentic RAG Chatbot")
st.markdown("This chatbot uses a knowledge base to answer your questions. Powered by LangGraph and Gemini.")

retriever = initialize_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

async def stream_response(graph, user_input, config):
    message = HumanMessage(content=user_input)
    async for event in graph.astream_events({"messages": [message]}, config, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, streaming=True)
    
    retriever_tool = Tool(
        name="document_retriever",
        description="Searches and returns relevant information and context from a knowledge base of documents.",
        func=retriever.invoke,
    )
    tools = [retriever_tool]
    llm_with_tools = llm.bind_tools(tools)
    
    from functools import partial
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

    with st.chat_message("assistant"):
        full_response = st.write_stream(stream_response(graph, prompt, config))
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
