import uvicorn
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

# Import the compiled LangGraph app from your agent.py
from app.agent import app

# Initialize the FastAPI app
api = FastAPI(
    title="Apparel Chatbot API",
    description="API for the multi-agent apparel customer service chatbot."
)


# Define the Pydantic models for our API's request and response
class InputChat(BaseModel):
    """The request model for the /chat endpoint."""
    query: str
    thread_id: str | None = None


class OutputChat(BaseModel):
    """The response model for the /chat endpoint."""
    response: str
    thread_id: str


# This is a synchronous endpoint
@api.post("/chat", response_model=OutputChat)
def chat(request: InputChat):
    """
    Main chat endpoint.

    Receives a query and an optional thread_id.
    Runs the query through the LangGraph agent.
    Returns the AI's response and the thread_id.
    """

    # If no thread_id is provided, create a new one
    thread_id = request.thread_id or str(uuid.uuid4())

    # Set up the config for the LangGraph agent
    config = {"configurable": {"thread_id": thread_id}}

    # Create the input message
    input_message = [HumanMessage(content=request.query)]

    final_response = None

    # Use the synchronous 'stream' method
    for event in app.stream(
            {"messages": input_message},
            config=config,
            stream_mode="values"
    ):
        new_messages = event["messages"]
        if new_messages:
            last_message = new_messages[-1]
            # We look for the last AIMessage that is NOT a tool call
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:

                # VVVV --- THIS IS THE FIX --- VVVV

                # Get the content from the final message
                raw_content = last_message.content

                if isinstance(raw_content, list) and raw_content:
                    # If it's a list (Gemini complex output),
                    # extract the text from the first part.
                    first_part = raw_content[0]
                    if "text" in first_part:
                        final_response = first_part["text"]
                elif isinstance(raw_content, str):
                    # If it's already a string (simple RAG output), use it.
                    final_response = raw_content

                # ^^^^ --- END OF FIX --- ^^^^

    if final_response is None:
        final_response = "Sorry, I couldn't find an answer to that."

    # This return will now succeed because final_response is a string
    return OutputChat(response=final_response, thread_id=thread_id)


# Add a simple root endpoint for testing
@api.get("/")
def root():
    """A simple health-check endpoint."""
    return {"status": "Apparel Chatbot API is running"}


# Main entry point to run the server
if __name__ == "__main__":
    print("Starting API server on http://127.0.0.1:8000")
    uvicorn.run("server:api", host="127.0.0.1", port=8000, reload=True)
