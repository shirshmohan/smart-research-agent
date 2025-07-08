import os
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from typing import List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Assuming these are in your tools directory and are correctly implemented
from tools.pdf_tool import load_and_summarize
from tools.search_tool import search_serpapi
from tools.rank_and_cite_tool import rank_and_cite


# Define a custom graph state that includes messages
class AgentState(dict):
    messages: Annotated[List[BaseMessage], add_messages]


if __name__ == "__main__":
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Prepare your tools list as before
    tools = [load_and_summarize, search_serpapi, rank_and_cite]

    # Create LangGraph React agent
    # Omit the 'prompt' argument to use the default ReAct prompt
    agent_runnable = create_react_agent(llm, tools)

    # Create state schema and graph
    graph = StateGraph(AgentState)

    # Define the agent node
    def call_agent(state: AgentState):
        # The prebuilt agent's prompt expects 'messages' as the input key for the conversation.
        # It handles breaking down the messages into input/chat_history internally.
        response = agent_runnable.invoke({"messages": state["messages"]})

        # The 'response' from agent_runnable.invoke() is a dictionary
        # representing the agent's internal state, which includes a 'messages' key
        # containing the updated list of messages (including tool calls, observations, and final answer).
        # We need to return *this list* of messages, not wrap the dictionary.
        return {"messages": response["messages"]} # <--- FIX: Extract 'messages' from the response dictionary

    graph.add_node("agent", call_agent)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)

    # Compile the graph
    app = graph.compile()


    # Check PDF path
    pdf_path = "C:/Users/KIIT0001/Downloads/BenefitsofGamingGranicLobelEngels2014.pdf"
    if Path(pdf_path).is_file():
        print("File found.")
    else:
        print("File not found.")

    # Queries to test
    query1_messages = [HumanMessage(content=f'Summarize the PDF file at "{pdf_path}"')]
    query2_messages = [HumanMessage(content="Search for benefits of going to the gym and rank the sources by credibility.")]

    # Invoke queries via LangGraph
    print("\nQuery 1 Result:\n")
    response1 = app.invoke({"messages": query1_messages})
    # The output will be in the 'messages' key of the returned state
    # The last message should be the AI's final answer, or potentially a ToolMessage if it called a tool and returned direct.
    # We want the content of the final AI message or tool message.
    # It's safest to iterate backwards to find the actual content.
    final_output_content = ""
    for msg in reversed(response1["messages"]):
        if hasattr(msg, 'content') and msg.content:
            final_output_content = msg.content
            break
    print(final_output_content)


    print("\nQuery 2 Result:\n")
    response2 = app.invoke({"messages": query2_messages})
    final_output_content = ""
    for msg in reversed(response2["messages"]):
        if hasattr(msg, 'content') and msg.content:
            final_output_content = msg.content
            break
    print(final_output_content)