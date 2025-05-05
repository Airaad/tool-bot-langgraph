from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from openai import OpenAI
from pydantic import BaseModel

import os

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMENI_API_KEY")

@tool
def run_command(command):
    """Takes a multiline command as input and executes a shell command and returns exit status."""
    result = os.system(command)
    return result

tools = [run_command]

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)
llm_with_tools = llm.bind_tools(tools = tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}



graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# Creates the graph without any memory
graph = graph_builder.compile()

# Creates graph with memory with the given checkpointer. This checkpointer can be MongoDB, Postgres etc.
def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
