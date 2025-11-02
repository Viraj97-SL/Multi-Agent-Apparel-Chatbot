import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()


def main():
    # 1. Check if API key is loaded
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found. Did you create your .env file?")
        return

    # 2. Check if LangSmith is configured
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    if not langsmith_key:
        print("Warning: LangSmith API key not found. Tracing will be disabled.")

    print("Environment setup is successful!")

    # 3. Test a simple LLM call
    try:
        # V-- THIS IS THE FIX --V
        # We need to use the full model name, "gemini-1.5-flash-latest"
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

        response = llm.invoke("Hello, world! Respond with one word.")
        print(f"LLM test successful. Response: {response.content}")
    except Exception as e:
        print(f"Error during LLM test: {e}")


if __name__ == "__main__":
    main()
