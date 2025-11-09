import os
import json
import sys
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, List
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
# MCP Adapter Imports
from langchain_mcp_adapters.client import MultiServerMCPClient
# --- Our Custom Imports ---
from app.chat_with_rag import create_rag_chain
# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
# Use AsyncSqliteSaver for async support
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# For supervisor tool-calling
from langchain_core.tools import tool
from pydantic import BaseModel, Field

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

# --- Agent 2: The Data Query Agent (now via MCP) ---
print("Initializing Data Query Tools via MCP...")
python_path = sys.executable  # Path to your Python executable

async def initialize_data_tools():
    client = MultiServerMCPClient(
        {
            "data_query": {
                "transport": "stdio",
                "command": python_path,
                "args": [os.path.join(os.path.dirname(__file__), "data_query_server.py")],  # The script as an argument
            }
        }
    )
    tools = await client.get_tools()  # Load MCP tools as LangChain tools
    return tools, client

# Run the async init and assign results
data_tools_list, mcp_client = asyncio.run(initialize_data_tools())
llm_with_data_tools = llm.bind_tools(data_tools_list)
data_tool_lookup = {t.name: t for t in data_tools_list}
print(f"Data Query tools initialized via MCP: {[t.name for t in data_tools_list]}")

# --- Agent 3: The Web Search Agent (unchanged) ---
print("Initializing Web Search Tool...")
web_search_tool = TavilySearchResults(max_results=3, name="tavily_general_search")
web_search_tools_list = [web_search_tool]
llm_with_web_search_tools = llm.bind_tools(web_search_tools_list)
web_search_tool_lookup = {t.name: t for t in web_search_tools_list}
print("Web Search tool initialized.")

# --- PROMPT for Data Query Agent ---
data_query_system_prompt = (
    """You are a helpful database assistant. You will be given a conversation history and a user's question. 
You have tools to query the product database, order database, and process returns. 

*** PROACTIVE SLOT-FILLING ***
If the user expresses an intent to use a tool (like 'initiate_return') but is missing required arguments (like 'order_id' or 'product_ids'), you must NOT call the tool. 
Instead, your response should be to ask the user for the *first* piece of missing information.

*** PROACTIVE OUT-OF-STOCK HANDLING ***
If a user asks for a product and the tool shows it's 'Out of Stock', you MUST be proactive: 
1. First, inform them it is out of stock. 
2. Then, *proactively ask them* if they would like to be added to the restock notification list. 
3. **Crucially, end your response with: "Please let me know."** This signals to the supervisor that you are expecting a "yes" or "no" follow-up.

Example Response: "It looks like the 'Rosalind Blossom Dress' in Medium is Out of Stock. Would you like me to notify you when it's back? Please let me know."

*** IMAGE HANDLING ***
If the tool's output includes one or more image tags (like '<img src="https..."/>'), you MUST include ALL image tags exactly as they appear, without modification, at the very end of your response.
"""
)

# --- 3. Define Graph Nodes (all async) ---
async def rag_agent_node(state: AgentState):
    """Calls the RAG chain for policy questions."""
    print("\n--- Calling Policy Agent ---")
    messages = state["messages"]
    last_human_message = messages[-1].content
    # Use .ainvoke for async RAG
    response_str = await rag_agent_chain.ainvoke(last_human_message)
    return {"messages": [AIMessage(content=response_str)]}

async def data_query_agent_node(state: AgentState):
    """Calls the LLM bound to the Data Query tools."""
    print("\n--- Calling Data Query Agent (LLM) ---")
    messages_with_prompt = [
        HumanMessage(content="SYSTEM INSTRUCTIONS - READ CAREFULLY"),
        AIMessage(content=data_query_system_prompt),
        *state["messages"]
    ]
    # Use .ainvoke for async
    ai_response = await llm_with_data_tools.ainvoke(messages_with_prompt)
    return {"messages": [ai_response]}

async def data_query_tool_executor_node(state: AgentState):
    """Executes the Data Query tools concurrently."""
    print("\n--- Calling Data Query Tool Executor ---")
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []
    if not last_message.tool_calls:
        return {"messages": []}
    tasks = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f" -> Preparing tool: {tool_name} with args {tool_args}")
        tool_to_call = data_tool_lookup.get(tool_name)
        if not tool_to_call:
            tasks.append((tool_call["id"], f"Error: Tool '{tool_name}' not found."))
        else:
            tasks.append(
                (tool_call["id"], tool_to_call.ainvoke(tool_args))
            )
    try:
        awaitable_tasks = [task for _, task in tasks if asyncio.iscoroutine(task)]
        results = await asyncio.gather(*awaitable_tasks)
        result_iter = iter(results)
        for i, (tool_call_id, task_or_error) in enumerate(tasks):
            if asyncio.iscoroutine(task_or_error):
                observation = next(result_iter)
            else:
                observation = task_or_error
            tool_messages.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call_id)
            )
    except Exception as e:
        print(f"Error during tool execution: {e}")
        for tool_call in last_message.tool_calls:
            tool_messages.append(
                ToolMessage(content=f"Error executing tool {tool_call['name']}: {e}", tool_call_id=tool_call["id"])
            )
    print("Tool Observations:", [msg.content for msg in tool_messages])
    return {"messages": tool_messages}

async def web_search_agent_node(state: AgentState):
    """Calls the LLM bound to the Web Search tools."""
    print("\n--- Calling Web Search Agent (LLM) ---")
    messages = state["messages"]
    ai_response = await llm_with_web_search_tools.ainvoke(messages)
    return {"messages": [ai_response]}

async def web_search_tool_executor_node(state: AgentState):
    """Executes the Web Search tools sequentially (since Tavily is sync)."""
    print("\n--- Calling Web Search Tool Executor ---")
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []
    if not last_message.tool_calls:
        return {"messages": []}
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f" -> Executing tool: {tool_name} with args {tool_args}")
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

# --- 4. Define the Supervisor (Router) with tool-calling for structured output ---
print("Initializing Supervisor...")
# Define Route tool
class RouteArgs(BaseModel):
    next_node: str = Field(..., enum=["rag_agent", "data_query_agent", "web_search_agent", "__end__"])

@tool("route", args_schema=RouteArgs)
def route(next_node: str) -> str:
    """Route to the next node."""
    return next_node

llm_with_route = llm.bind_tools([route])

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are the 'supervisor' of an apparel store customer service system. Your job is to route the user's most recent message to the correct specialist agent. You have three specialists: 
1. 'rag_agent': The *Policy Agent*. Use for questions about store policies (shipping, returns, T&Cs). 
2. 'data_query_agent': The *Database Agent*. Use for questions about products (price, size, stock) AND orders (status, returns). 
3. 'web_search_agent': The *Public Web Searcher*. Use as a 'catch-all' for fashion trends, company news, etc. 

**Follow these steps to make your decision:**
1. Examine the user's *most recent message*.
2. Examine the AI's *immediately preceding message*.
3. **Check for a follow-up:** If the AI's last message was a *question* (e.g., "...what is your order ID?", "...would you like to be notified?", "...what is your email?"), and the user's message looks like an *answer* (e.g., "yes please", "my email is test@example.com", "ORD-123"), then the conversation is *continuing*. You MUST route to the **'data_query_agent'**.
4. **Check for a greeting/ending:** If it's *not* a follow-up, check if the user's message is a simple greeting, "thank you," or "goodbye." If so, you MUST route to **'__end__'**.
5. **Check for a new task:** If it's *not* a follow-up or an ending, it's a new task. Route it to the correct specialist: 
   - 'rag_agent': For store policies (shipping, returns). 
   - 'data_query_agent': For products (price, stock) or orders. 
   - 'web_search_agent': For fashion trends, company news, etc. 

You MUST call the 'route' tool with the next_node argument set to one of: 'rag_agent', 'data_query_agent', 'web_search_agent', or '__end__'.
Do not output anything else."""),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

async def supervisor_router(state: AgentState):
    """Routes the conversation to the appropriate agent."""
    print("\n--- Supervisor Routing ---")
    messages = state["messages"]
    ai_response = await (supervisor_prompt | llm_with_route).ainvoke({"messages": messages})
    if ai_response.tool_calls:
        next_node = ai_response.tool_calls[0]["args"]["next_node"]
    else:
        next_node = "__end__"  # Fallback if no tool call
    print(f"Supervisor decided: -> {next_node}")
    return {"next": next_node}

# --- 5. Define Conditional Edges ---
def check_for_tool_calls(state: AgentState) -> str:
    """If the agent called tools, run the executor. Otherwise, loop back to supervisor."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue_with_tools"
    else:
        # If the agent *spoke* (no tool call), we loop back to the supervisor
        # to wait for the next human input.
        return "supervisor"

# --- 6. Build the Graph ---
print("Building graph...")
workflow = StateGraph(AgentState)
# Add all nodes (use async versions)
workflow.add_node("supervisor", supervisor_router)
workflow.add_node("rag_agent", rag_agent_node)
workflow.add_node("data_query_agent", data_query_agent_node)
workflow.add_node("data_query_tool_executor", data_query_tool_executor_node)
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
        "data_query_agent": "data_query_agent",
        "web_search_agent": "web_search_agent",
        "__end__": END,
    },
)

# RAG agent edge
workflow.add_conditional_edges(
    "rag_agent",
    check_for_tool_calls, # This will always return "supervisor"
    {
        "continue_with_tools": "supervisor", # Fallback
        "supervisor": "supervisor"
    }
)

# Data query edges
workflow.add_conditional_edges(
    "data_query_agent",
    check_for_tool_calls,
    {
        "continue_with_tools": "data_query_tool_executor",
        "supervisor": "supervisor", # <-- Agent now loops back
    },
)
workflow.add_edge("data_query_tool_executor", "data_query_agent")

# Web Search agent edges
workflow.add_conditional_edges(
    "web_search_agent",
    check_for_tool_calls,
    {
        "continue_with_tools": "web_search_tool_executor",
        "supervisor": "supervisor", # <-- Agent now loops back
    },
)
workflow.add_edge("web_search_tool_executor", "web_search_agent")

# --- 7. Compile the Graph and Set Up Memory ---
async def create_memory():
    # Use a file-based DB for persistence across server restarts
    conn = await aiosqlite.connect("checkpoints.db")
    return AsyncSqliteSaver(conn=conn)

memory = asyncio.run(create_memory())
app = workflow.compile(checkpointer=memory)
print("\n--- Graph Compiled Successfully! ---")

# --- 8. Run the Chatbot (Handled by server.py) ---
