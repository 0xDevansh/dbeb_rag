import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyCajA-t68BWLwRSrc3qolSBNRGw70xMnzo"
os.environ["HF_HUB_ENABLE_SYMLINKS"] = "1"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

import torch
import logging
logging.getLogger('google.api_core').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)

import pprint
import sqlite3
import asyncio
import json
import uuid
from IPython.display import Image, display
from datetime import datetime

from typing import Dict, List, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass, field
from enum import Enum

from typing import Dict, List, Any, TypedDict, Literal

from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langchain_community.storage import SQLStore
# from langchain_community.storage.sql import SQLStore
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser
# from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser

from langchain_core.tools import Tool
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AnyMessage, SystemMessage, AIMessageChunk
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic _v1import BaseModel, Field

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

if torch.cuda.is_available():
    device = 'cuda'
    print("✅ CUDA is available. Using GPU for embeddings.")
else:
    device = 'cpu'
    print("⚠️ CUDA not available. Using CPU for embeddings. This may be slow.")

model_kwargs = {
    'device': device
}

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs=model_kwargs
)

url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

qdrant = QdrantVectorStore(
    client=client,
    collection_name="dbeb", # The name of your existing collection
    embedding=embeddings,
)

retriever = qdrant.as_retriever()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

class State(MessagesState):
    pass

retriever_tool = Tool(
    name="document_retriever",
    description="Searches and returns relevant information and context from a knowledge base of documents.",
    func=retriever.invoke,
)

tools = [retriever_tool]
llm_with_tools = llm.bind_tools(tools)

async def agent_node(state: State):
    """
    The agent node that decides whether to call a tool or respond directly.
    """
    print("NODE: AGENT")
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {
        "messages": [response]
    }

tool_node = ToolNode(tools=tools)

async def run_chatbot(graph):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("\n\n🤖 Agentic RAG Chatbot is ready!")
    print(f"Session ID: {thread_id}")
    print("Ask your questions. Type 'exit' to quit.")
    
    while True:
        # Use asyncio.to_thread to run the blocking input() in an async-friendly way
        user_input = await asyncio.to_thread(input, "\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye! 👋")
            break
            
        message = HumanMessage(content=user_input)
        
        try:
            print("Bot: ", end="", flush=True)
            
            # Use astream_events with version "v2" for the most detailed event stream.
            async for event in graph.astream_events({"messages": [message]}, config, version="v2"):
                kind = event["event"]
                
                # This is the canonical way to listen for token streams.
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if isinstance(chunk, AIMessageChunk) and not chunk.tool_call_chunks:
                        print(chunk.content, end="", flush=True)
            
            print() # Final newline
            
        except Exception as e:
            print(f"\n\n[ERROR] An error occurred: {e}")
            print("Please try your query again.")


# 5. Build and Compile the Graph, and run with asyncio
async def main():
    print("Building the agentic graph...")
    # AsyncSqliteSaver requires an async context manager
    async with AsyncSqliteSaver.from_conn_string("chatbot_memory.sqlite") as memory:
        workflow = StateGraph(State)

        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            tools_condition,
        )
        
        workflow.add_edge("tools", "agent")

        graph = workflow.compile(checkpointer=memory)
        print("✅ Agentic graph compiled successfully.")

        # Run the chatbot from within the async context
        await run_chatbot(graph)

if __name__ == "__main__":
    # Add nest_asyncio to patch the event loop for environments like Jupyter
    import nest_asyncio
    nest_asyncio.apply()

    # Run the main async function
    asyncio.run(main())
