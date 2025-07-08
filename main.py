import os
from dotenv import load_dotenv
from typing import List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()
os.getenv("OPENAI_API_KEY")


from tools.pdf_tool import load_and_summarize
from tools.search_tool import search_serpapi
from tools.rank_and_cite_tool import rank_and_cite
from tools.pdf_compare import compare_documents

# Define a custom graph state that includes messages
# The 'add_messages' annotator handles appending new messages to the list
class AgentState(dict):
    messages: Annotated[List[BaseMessage], add_messages]


if __name__ == "__main__":
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Prepare your tools list
    tools = [load_and_summarize, search_serpapi, rank_and_cite, compare_documents]

    # Initialize Embedding and ChromaDB
    # Note: In this version, the `vectorstore` is initialized but not directly
    # used for the agent's conversational memory persistence.
    # It might be used by your tools, though.
    persist_directory = "./chroma_memory"
    embedding = OpenAIEmbeddings()

    vectorstore = Chroma(
        collection_name="chat_history",
        embedding_function=embedding,
        persist_directory=persist_directory
    )

    # Create the LangGraph React agent
    # `create_react_agent` from `langgraph.prebuilt` does not take a 'memory' argument.
    # Its memory is handled by the `AgentState` and `add_messages`.
    agent_runnable = create_react_agent(llm, tools)

    # Define the LangGraph StateGraph
    graph = StateGraph(AgentState)

    # Define the node that calls your agent
    def call_agent(state: AgentState):
        # The prebuilt agent expects 'messages' as the input key for the current conversation.
        # It returns a dictionary that includes the updated 'messages'.
        response = agent_runnable.invoke({"messages": state["messages"]})
        # Return the updated messages to be added to the graph's state by 'add_messages'
        return {"messages": response["messages"]}

    # Add the agent node to the graph
    graph.add_node("agent", call_agent)
    # Set the entry point of the graph to the agent node
    graph.set_entry_point("agent")
    # Define an edge from the agent node to the END point (simple, one-step graph)
    graph.add_edge("agent", END)

    # Compile the graph
    # No checkpointer is used here, so memory is not persistent across runs.
    app = graph.compile()

    # Main loop for interaction
    print("Starting conversation. Type 'exit' to quit.")
    user_input = input("\nHi!I am an AI Research agent.Please enter  your query: ")
    while True:
        
        if user_input.lower() in ['exit', "quit"]:
            break

        # For each new user input, start with a HumanMessage.
        # LangGraph's 'add_messages' will append this to the existing state.
        current_messages = [HumanMessage(content=user_input)]
        response = app.invoke({"messages":current_messages})
        agent_response_message = response["messages"][-1]
        print("Agent- ",agent_response_message.content)
        print("Anything else I can help you with?If not type 'exit' ")
