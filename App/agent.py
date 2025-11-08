import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, List
import operator
import sqlite3

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Our Custom Imports ---
from app.chat_with_rag import create_rag_chain
# --- RENAMED TOOL IMPORT ---
from app.tools import get_data_query_tools

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env file.")


# --- 1. Define Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# --- 2. Define Agents and Tools ---

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# --- Agent 1: The Policy Agent (RAG) ---
print("Initializing Policy Agent (RAG)...")
rag_agent_chain = create_rag_chain()
if rag_agent_chain is None:
    print("Error: RAG chain (Policy Agent) could not be created.")
    exit()
print("Policy Agent initialized.")

# --- Agent 2: The Data Query Agent (SQL Tools) ---
print("Initializing Data Query Tools (Products, Orders, Returns)...")
# --- USE NEW FUNCTION & RENAMED VARIABLES ---
data_tools_list = get_data_query_tools()
llm_with_data_tools = llm.bind_tools(data_tools_list)
data_tool_lookup = {t.name: t for t in data_tools_list}
print(f"Data Query tools initialized: {[t.name for t in data_tools_list]}")

# --- Agent 3: The Web Search Agent ---
print("Initializing Web Search Tool...")
web_search_tool = TavilySearchResults(max_results=3, name="tavily_general_search")
web_search_tools_list = [web_search_tool]
llm_with_web_search_tools = llm.bind_tools(web_search_tools_list)
web_search_tool_lookup = {t.name: t for t in web_search_tools_list}
print("Web Search tool initialized.")

#
# VVVV --- THIS IS THE NEW PROMPT (CHANGE 1) --- VVVV
#
# --- NEW PROMPT for Data Query Agent ---
# This prompt instructs the LLM on how to behave
# when it's summarizing tool output.
data_query_system_prompt = (
    """You are a helpful database assistant. You will be given a conversation history and a user's question. 
You have tools to query the product database, order database, and process returns. 
When a tool provides you with information, summarize it clearly for the user in a natural, concise paragraph.

*** IMPORTANT ***
If the tool's output includes one or more image tags (like '<img src="https://example.com/image.jpg" alt="Image" />'), 
you MUST include ALL image tags exactly as they appear, without modification, at the very end of your response. 
Do not omit, rephrase, or wrap them—copy them verbatim. Place them after your summary text, separated by a newline if needed.
This ensures the images are displayed properly in the final output.

Example:
Tool Output: "Product: Red Dress\nCategory: DRESS\nColour: RED\nPrice: 50.0\n<img src=\"https://example.com/red-dress.jpg\" alt=\"Image\" />\n\nProduct: Blue Shirt\nCategory: SHIRT\nColour: BLUE\nPrice: 30.0\n<img src=\"https://example.com/blue-shirt.jpg\" alt=\"Image\" />"
Your Response: "The Red Dress is in the DRESS category, available in RED, and costs 50.0. The Blue Shirt is in the SHIRT category, available in BLUE, and costs 30.0.\n<img src=\"https://example.com/red-dress.jpg\" alt=\"Image\" />\n<img src=\"https://example.com/blue-shirt.jpg\" alt=\"Image\" />"
"""
)

# --- 3. Define Graph Nodes ---

def rag_agent_node(state: AgentState):
    """Calls the RAG chain for policy questions."""
    print("\n--- Calling Policy Agent ---")
    messages = state["messages"]
    last_human_message = messages[-1].content
    response_str = rag_agent_chain.invoke(last_human_message)
    return {"messages": [AIMessage(content=response_str)]}


#
# VVVV --- THIS IS THE UPDATED FUNCTION (CHANGE 2) --- VVVV
#
# --- RENAMED DATA QUERY NODES ---
def data_query_agent_node(state: AgentState):
    """Calls the LLM bound to the Data Query tools."""
    print("\n--- Calling Data Query Agent (LLM) ---")

    # We "pre-prompt" the agent by adding our system instructions
    # before the rest of the conversation history.

    # We use a Human/AI pair to simulate a "system" prompt for Gemini
    messages_with_prompt = [
        HumanMessage(content="SYSTEM INSTRUCTIONS - READ CAREFULLY"),
        AIMessage(content=data_query_system_prompt),
        *state["messages"]  # Add all the real messages after
    ]

    ai_response = llm_with_data_tools.invoke(messages_with_prompt)
    return {"messages": [ai_response]}


#
# ^^^^ --- END OF UPDATED FUNCTION --- ^^^^
#

def data_query_tool_executor_node(state: AgentState):
    """Executes the Data Query tools."""
    print("\n--- Calling Data Query Tool Executor ---")
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []

    if not last_message.tool_calls:
        return {"messages": []}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"  -> Executing tool: {tool_name} with args {tool_args}")
        # Use the data_tool_lookup
        tool_to_call = data_tool_lookup.get(tool_name)

        if not tool_to_call:
            observation = f"Error: Tool '{tool_name}' not found."
        else:
            try:
                observation = tool_to_call.invoke(tool_args)
            except Exception as e:
                observation = f"Error executing tool {tool_name}: {e}"

        tool_messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

    return {"messages": tool_messages}


# --- WEB SEARCH NODES (Unchanged) ---
def web_search_agent_node(state: AgentState):
    """Calls the LLM bound to the Web Search tools."""
    print("\n--- Calling Web Search Agent (LLM) ---")
    messages = state["messages"]
    ai_response = llm_with_web_search_tools.invoke(messages)
    return {"messages": [ai_response]}


def web_search_tool_executor_node(state: AgentState):
    """Executes the Web Search tools."""
    print("\n--- Calling Web Search Tool Executor ---")
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []

    if not last_message.tool_calls:
        return {"messages": []}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"  -> Executing tool: {tool_name} with args {tool_args}")
        tool_to_call = web_search_tool_lookup.get(tool_name)

        if not tool_to_call:
            observation = f"Error: Tool '{tool_name}' not found."
        else:
            try:
                observation = tool_to_call.invoke(tool_args)
            except Exception as e:
                observation = f"Error executing tool {tool_name}: {e}"

        tool_messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

    return {"messages": tool_messages}


# --- 4. Define the Supervisor (Router) ---
print("Initializing Supervisor...")
# --- SUPERVISOR PROMPT IS UPDATED ---
supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are the 'supervisor' of an apparel store customer service system. 
Your job is to route the user's most recent message to the correct specialist agent.
You have three specialists:

1. 'rag_agent': This is the *Policy Agent*.
   - Use it *only* for questions about store policies (shipping, returns, T&Cs, etc.).

2. 'data_query_agent': This is the *Database Agent*.
   - Use it for *all* questions about product details (price, size, color, description, image).
   - Use it for *all* questions about a *specific* order (e.g., "Where is order ORD-123?").
   - Use it for *all* requests to initiate a return.

3. 'web_search_agent': This is the *Public Web Searcher*.
   - Use it as a 'catch-all' for any general question that the other agents cannot answer.
   - Use it for questions about fashion trends, general knowledge, or company news.

Based on the user's last message, choose the next step. 
You must respond with *only* the name of the next node:

- 'rag_agent'
- 'data_query_agent'
- 'web_search_agent'
- '__end__' (if the conversation seems finished or is a simple greeting)"""),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def supervisor_router(state: AgentState):
    """Routes the conversation to the appropriate agent."""
    print("\n--- Supervisor Routing ---")
    messages = state["messages"]

    ai_response = (supervisor_prompt | llm).invoke({"messages": messages})
    content = ai_response.content

    if isinstance(content, list) and content:
        content = content[0].get("text", "")
    elif not isinstance(content, str):
        content = "__end__"

    next_node = content.strip().replace("'", "").replace('"', '')

    if not next_node:
        next_node = "__end__"

    print(f"Supervisor decided: -> {next_node}")
    return {"next": next_node}


# --- 5. Define Conditional Edges (Unchanged) ---

def check_for_tool_calls(state: AgentState) -> str:
    """Checks if the last message from an agent node contained tool calls."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue_with_tools"
    else:
        return "__end__"

    # --- 6. Build the Graph ---


print("Building graph...")
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("supervisor", supervisor_router)
workflow.add_node("rag_agent", rag_agent_node)
# --- RENAMED NODES ---
workflow.add_node("data_query_agent", data_query_agent_node)
workflow.add_node("data_query_tool_executor", data_query_tool_executor_node)
# ---
workflow.add_node("web_search_agent", web_search_agent_node)
workflow.add_node("web_search_tool_executor", web_search_tool_executor_node)

# Set the entry point
workflow.set_entry_point("supervisor")

# Supervisor conditional edges
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "rag_agent": "rag_agent",
        # --- RENAMED EDGE ---
        "data_query_agent": "data_query_agent",
        "web_search_agent": "web_search_agent",
        "__end__": END,
    },
)

# RAG agent edge
workflow.add_edge("rag_agent", END)

# --- RENAMED DATA QUERY EDGES ---
workflow.add_conditional_edges(
    "data_query_agent",
    check_for_tool_calls,
    {
        "continue_with_tools": "data_query_tool_executor",
        "__end__": END,
    },
)
workflow.add_edge("data_query_tool_executor", "data_query_agent")

# Web Search agent edges
workflow.add_conditional_edges(
    "web_search_agent",
    check_for_tool_calls,
    {
        "continue_with_tools": "web_search_tool_executor",
        "__end__": END,
    },
)
workflow.add_edge("web_search_tool_executor", "web_search_agent")

# --- 7. Compile the Graph and Set Up Memory (Unchanged) ---

conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn=conn)

app = workflow.compile(checkpointer=memory)
print("\n--- Graph Compiled Successfully! ---")

# --- 8. Run the Chatbot (Handled by server.py) ---
