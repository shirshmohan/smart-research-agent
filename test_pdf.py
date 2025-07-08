from tools.pdf_tool import load_and_summarize
from dotenv import load_dotenv
load_dotenv()
summary = load_and_summarize.invoke("C:/Users/KIIT0001/Documents/Sem4/java ppt/UNIT 1 (1).pdf")
print(summary)