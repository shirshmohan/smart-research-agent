# test_search.py
from dotenv import load_dotenv
load_dotenv()

from tools.search_tool import search_serpapi

query = "Top 5 Movies of the recent decade "
result = search_serpapi.run(query)

print("Search Results:\n\n")
print(result)
