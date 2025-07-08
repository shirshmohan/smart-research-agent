from tools.pdf_compare import compare_documents

# Provide paths to 2-3 local PDFs for testing
file_paths = [
    r"C:\Users\KIIT0001\Documents\end sem\LiteratureReview2305245.pdf",
r"C:\Users\KIIT0001\Documents\end sem\LiteratureReview2305245.pdf"
]

response = compare_documents.invoke({"file_paths":file_paths})
print("\n🧠 Document Comparison Result:\n")
print(response)
