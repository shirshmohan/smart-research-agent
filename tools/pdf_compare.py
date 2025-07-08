from langchain.tools import tool
from PyPDF2 import PdfReader
from typing import List
from langchain_openai import ChatOpenAI

llm=ChatOpenAI(model="gpt-3.5-turbo",temperature=0)

MAX_WORDS=500


def extract_text_from_pdf(file_path:str)->str:
    try:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text()) or ""
        words = text.split()
        limited_text = " ".join(words[:MAX_WORDS])

        return limited_text
    except Exception as e:
        return [f"Error {file_path}:{e}"]

@tool("compare_documents",return_direct=True)
def compare_documents(file_paths:List[str])->str:
    """
    Compares up to 5 documents and extract common,unqiue,and conflicting points.
    Input: List of filepaths to PDF documets
    Output: A summary highlighting same points,unique points,and contradictions.
    """
    if(len(file_paths)>10):
        return "Error :Max no. of 10 documents allowed."
    
    docs={}
    for path in file_paths:
        docs[path] = extract_text_from_pdf(path)
        input_prompt = """
        You are a smart document comparison assistant.
        You are given multiple documents.Your job is to:
        1.Identify common points.
        2.Identify points that are unique to each document.
        3.Identify any conflicting or contradictory points among the documents.

        Respond with a structured output under the following header:
        - Common Points:(Ex-"Document 1 and Document to suggesst that -")
        -Unique Points:(Ex-"Only document one suggests that )
        - Conflicting Points:

        Documents:

        """
        for i,(path,text) in enumerate(docs.items(),1):
            input_prompt += f"\nDocument {i} ({path}):\n{path}:\n {text}\n"

        response = llm.invoke(input_prompt)
        return response.content