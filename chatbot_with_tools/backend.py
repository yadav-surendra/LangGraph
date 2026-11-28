
# backend.py

from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from ddgs import DDGS
from dotenv import load_dotenv
import sqlite3
import requests
import os

load_dotenv()

# -------------------
# 1. LLM
# -------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# -------------------
# 2. Tools
# -------------------

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information.
    Use this tool when the user asks questions requiring
    online information or recent facts.
    """

    try:
        results = DDGS().text(
            query,
            region="us-en",
            safesearch="moderate",
            max_results=5
        )

        if not results:
            return "No search results found."

        formatted_results = []

        for result in results:
            formatted_results.append(
                f"Title: {result.get('title', 'N/A')}\n"
                f"URL: {result.get('href', 'N/A')}\n"
                f"Description: {result.get('body', 'N/A')}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Perform basic arithmetic operations.
    Supported operations: add, sub, mul, div.
    """

    try:
        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}

            result = first_num / second_num

        else:
            return {"error": f"Unsupported operation: {operation}"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest available stock quote using Alpha Vantage.
    Example symbols: AAPL, TSLA, MSFT.
    """

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        return {"error": "Alpha Vantage API key is missing."}

    try:
        url = "https://www.alphavantage.co/query"

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol.upper(),
            "apikey": api_key
        }

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        if "Error Message" in data:
            return {"error": data["Error Message"]}

        if "Note" in data or "Information" in data:
            return {
                "error": data.get("Note", data.get("Information"))
            }

        quote = data.get("Global Quote", {})

        if not quote:
            return {"error": f"No stock data found for {symbol}"}

        return {
            "symbol": quote.get("01. symbol"),
            "price": quote.get("05. price"),
            "latest_trading_day": quote.get("07. latest trading day"),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent")
        }

    except requests.RequestException as e:
        return {"error": f"Stock API request failed: {str(e)}"}

    except Exception as e:
        return {"error": str(e)}


# Register tools

tools = [
    web_search,
    get_stock_price,
    calculator
]

llm_with_tools = llm.bind_tools(tools)


# -------------------
# 3. State
# -------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 4. Nodes
# -------------------

def chat_node(state: ChatState):
    """LLM node that can answer or request tool execution."""

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


tool_node = ToolNode(tools)


# -------------------
# 5. Checkpointer
# -------------------

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn=conn)


# -------------------
# 6. Graph
# -------------------

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(
    checkpointer=checkpointer
)


# -------------------
# 7. Helper Functions
# -------------------

def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        all_threads.add(
            checkpoint.config["configurable"]["thread_id"]
        )

    return list(all_threads)
