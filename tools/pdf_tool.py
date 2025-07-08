from langchain.tools import tool
import fitz 
from dotenv import load_dotenv
import os 
from langchain_openai import ChatOpenAI
from transformers import pipeline

load_dotenv()

summarizer = pipeline("summarization",model="facebook/bart-large-cnn")

@tool("summarize-pdf",return_direct=True)
def load_and_summarize(input: str)->str:
    """Load and summarize a PDF from a local file path"""
    file_path = input.strip().strip('"')
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        if not text.strip():
            return "PDF appears to be empty or unreadable"   

        #llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)
        chunk = text[:1024]
        if len(chunk.strip().split())<50:
            return "Not enough content to summarize."
        summary = summarizer(chunk,max_length=200,min_length=50,do_sample=False)
        return summary[0]['summary_text']

        #prompt = f"Summarize the following PDF content in 5-6 bullet points: \n\n{text}"
        #summary = llm.invoke(prompt)

        #return summary.content if hasattr(summary,'content') else summary
    except Exception as e:
        return f"Error summarizing PDF: {str(e)}"