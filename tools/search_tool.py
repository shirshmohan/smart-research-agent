from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from serpapi import GoogleSearch
from sentence_transformers import CrossEncoder
import os

ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@tool("search_web",return_direct=True)
def search_serpapi(query:str)-> str:
    """Searches the web using SerpAPI and returns top 5 result snippets."""
    try:
        params = {
            "engine":"google",
            "q":query,
            "api_key":os.getenv("SERPAPI_API_KEY"),
            "num":10,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        if "error" in results:
            return f"Error:{results['error']}"
        if "organic_results" not in results:
            return "No search results found"
        
        entries = results["organic_results"]
        scored = [
            (entry.get("snippet", ""), entry.get("link", ""), float(ranker.predict([(query, entry.get("snippet", ""))])))
            for entry in entries
            if "snippet" in entry and "link" in entry
        ]
        ranked = sorted(scored, key=lambda x: x[2], reverse=True)
        

        top_results = ranked[:5]

        summary = "Here's what I found:\n"
        for i, (snippet, url, score) in enumerate(top_results, start=1):
            summary += f"{i}. {snippet}\n   🔗 Source: {url} (Relevance: {score:.2f})\n"
        return summary
    except Exception as e:
        return f"Search failed:{str(e)}"
