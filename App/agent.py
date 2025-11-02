import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, List
import operator
import sqlite3

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Our Custom Imports ---
from app.chat_with_rag import create_rag_chain
from app.tools import get_all_tools

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")


# --- 1. Define Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# --- 2. Define Agents and Tools ---

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

print("Initializing Tools...")
all_tools_list = get_all_tools()
print(f"Tools initialized: {[t.name for t in all_tools_list]}")

llm_with_tools = llm.bind_tools(all_tools_list)
tool_lookup = {t.name: t for t in all_tools_list}

print("Initializing RAG Agent...")
rag_chain = create_rag_chain()
if rag_chain is None:
    print("Error: RAG chain could not be created. Please check 'chat_with_rag.py'.")
    exit()
print("RAG Agent initialized.")


# --- 3. Define Graph Nodes ---

def tool_agent_node(state: AgentState):
    """Calls the LLM. If it decides to use tools, it will output tool_calls."""
    print("\n--- Calling Tool Agent (LLM) ---")
    messages = state["messages"]
    ai_response = llm_with_tools.invoke(messages)
    return {"messages": [ai_response]}


def tool_executor_node(state: AgentState):
    """Executes the tool calls decided by the 'tool_agent_node'."""
    print("\n--- Calling Tool Executor ---")
    messages = state["messages"]
    last_message = messages[-1]

    tool_messages = []

    if not last_message.tool_calls:
        print("No tool calls found.")
        return {"messages": []}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"  -> Executing tool: {tool_name} with args {tool_args}")
        tool_to_call = tool_lookup.get(tool_name)

        if not tool_to_call:
            print(f"Error: Tool '{tool_name}' not found.")
            observation = f"Error: Tool '{tool_name}' not found."
        else:
            try:
                observation = tool_to_call.invoke(tool_args)
            except Exception as e:
                print(f"Error executing tool: {e}")
                observation = f"Error executing tool {tool_name}: {e}"

        tool_messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )
    return {"messages": tool_messages}


def rag_agent_node(state: AgentState):
    """Calls the RAG chain for general knowledge questions."""
    print("\n--- Calling RAG Agent ---")
    messages = state["messages"]
    last_human_message = messages[-1].content
    response_str = rag_chain.invoke(last_human_message)
    return {"messages": [AIMessage(content=response_str)]}


# --- 4. Define the Supervisor (Router) ---
print("Initializing Supervisor...")
supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are the 'supervisor' of an apparel store customer service system. 
Your job is to route the user's most recent message to the correct specialist 
or to end the conversation. You have two specialists:

1. 'tool_agent': This agent handles specific, live data tasks. 
   - Use it for questions about order status, tracking, or shipping. 
   - Use it for requests to initiate returns. 
   - Use it for any questions involving specific order IDs (e.g., 'ORD-123').
   - Use it if the user asks what's in their order or to find customer details.

2. 'rag_agent': This agent is a knowledge base expert. 
   - Use it for general questions about products (e.g., 'What colors?'). 
   - Use it for questions about store policies (shipping, returns). 
   - Use it for any general questions that do NOT involve a specific order ID.

If the user is saying 'hello', 'thanks', or 'goodbye', you can just respond directly.

Based on the user's last message, choose the next step. 
You must respond with *only* the name of the next node:

- 'tool_agent'
- 'rag_agent'
- '__end__' (if the conversation seems finished or is a simple greeting)"""),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# This node function is correct: it returns a DICTIONARY to update the state.
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

    # VVV THIS IS FIX #1: Return a dictionary to update the 'next' key
    return {"next": next_node}


# --- 5. Define Conditional Edges ---

def should_continue(state: AgentState) -> str:
    """Decides if the tool agent should run again or end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "continue_tool_executor"
    else:
        return "__end__"

    # --- 6. Build the Graph ---


print("Building graph...")
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_router)
workflow.add_node("tool_agent", tool_agent_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("rag_agent", rag_agent_node)

workflow.set_entry_point("supervisor")

# VVV THIS IS FIX #2: The conditional edge function now READS
# the 'next' key from the state to get the string key.
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "tool_agent": "tool_agent",
        "rag_agent": "rag_agent",
        "__end__": END,
    },
)

# This conditional edge is correct
workflow.add_conditional_edges(
    "tool_agent",
    should_continue,
    {
        "continue_tool_executor": "tool_executor",
        "__end__": END,
    },
)

# These edges are correct
workflow.add_edge("rag_agent", END)
workflow.add_edge("tool_executor", "tool_agent")

# --- 7. Compile the Graph and Set Up Memory ---

conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn=conn)

app = workflow.compile(checkpointer=memory)
print("\n--- Graph Compiled Successfully! ---")

# --- 8. Run the Chatbot ---

if __name__ == "__main__":
    print("--- Apparel Agent is Ready ---")
    print("Ask questions about products, policies, or orders. Type 'exit' to quit.")

    import uuid

    thread_id = str(uuid.uuid4())
    print(f"Using conversation ID: {thread_id}")

    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue

            config = {"configurable": {"thread_id": thread_id}}
            input_message = [HumanMessage(content=query)]

            print("Bot:")
            final_response = None
            for event in app.stream({"messages": input_message}, config=config, stream_mode="values"):
                new_messages = event["messages"]

                if new_messages:
                    last_message = new_messages[-1]
                    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                        final_response = last_message.content

            if final_response:
                print(f"-> {final_response}")
            else:
                # Handle cases where the graph ends without a final AIMessage
                # This can happen if the tool agent errors out
                if not final_response and state["messages"]:
                    # Try to print the last message if it's an error
                    last_msg_content = state["messages"][-1].content
                    if "Error:" in str(last_msg_content):
                        print(f"-> An error occurred: {last_msg_content}")
                    else:
                        print("-> (Graph finished execution)")
                else:
                    print("-> (Graph finished execution)")


        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break
