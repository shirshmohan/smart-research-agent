# 🧠 Smart Research Agent (LangGraph)

An intelligent agent that can summarize PDFs, search the web, rank sources by credibility, and more — built using LangGraph, LangChain tools, and OpenAI.

## Features
- 📄 PDF Summarization
- 🌐 Web Search + Ranking
- 🧠 LLM-Powered Tool Usage
-  📃 Document comparer
- 🧰 Tool chaining via LangGraph
- - 🧠 **Session Memory (ChromaDB-based):** Stores chat history for each session using vector embeddings, enabling context-aware conversations.


## Setup

1. Create a `.env` file:
    ```env
    OPENAI_API_KEY=your-openai-key
    SERPAPI_API_KEY=your-serpapi-key
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run:
    ```bash
    python main.py
    ```

## Upcoming
- 📺 YouTube summarizer
- 📊 CSV analyzer
- 🖥️ React + FastAPI frontend
